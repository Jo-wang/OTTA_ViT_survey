
import random
import os

import torch
import torch.utils.data as data
import torchvision.utils as tvu
import torch.nn.functional as F
import networks.augmentation.classification_policies as policy

from tqdm import tqdm
from dataloaders.classification import get_dataset
from networks.classification.utils import build_network
from networks.augmentation.optaug import OptAug
from losses.classification_losses import EntropyClassMarginals, EntropyLoss
from utilities.distributed import DistributedEvalSampler, concat_all_gather, model_to_DDP, sync_batchnorms
from utilities.metadata_tracker import MetadataTracker
from utilities.metric_tracker import MetricTracker
from utilities.utils import ensure_dir, normalize_fn, denormalize_fn, offline
from networks.ema import ModelEMA


class TeSLA(object):
    def __init__(self, opt):

        # Setup attributes
        opt.aug_mult = 1
        self.opt = opt

        # initialize data loaders
        self.target_dataset = get_dataset(opt.dataset_name, opt.target_data_path, opt,
                                          aug_mult=opt.aug_mult_easy, hard_augment=opt.hard_augment)

        # Parallel processing
        if self.opt.dist:
            sampler = DistributedEvalSampler(self.target_dataset, shuffle=self.opt.n_epochs>1)
            shuffle = False
            pin_memory=False
        else:
            sampler = None
            shuffle = True
            pin_memory=False

        self.target_loader = data.DataLoader(self.target_dataset, batch_size=self.opt.batch_size, sampler=sampler,
                                            pin_memory=pin_memory, shuffle=shuffle, drop_last=False, num_workers=opt.num_workers)
        self.eval_target_loader = data.DataLoader(self.target_dataset, batch_size=self.opt.batch_size*2, sampler=sampler,
                                            pin_memory=pin_memory, shuffle=shuffle, drop_last=False, num_workers=opt.num_workers)

        # source model
        self.net = build_network(opt.arch, opt.n_classes, opt.pretrained_source_path)
        self.net = self.net.cuda()
        self.set_parameters(self.net)
        self.softmax = torch.nn.Softmax(dim=-1)

        # Load source stats if we load them
        if self.opt.use_source_stats:
            self.feat_dim = self.net.dim
            self.source_dataset = get_dataset(opt.dataset_name, opt.source_data_path, opt, load_source=True, aug_mult=0)

            if self.opt.dist:
                sampler = DistributedEvalSampler(self.source_dataset, shuffle=False)

            self.source_loader = data.DataLoader(self.source_dataset, batch_size=self.opt.batch_size, sampler=sampler,
                                                pin_memory=pin_memory, shuffle=shuffle, drop_last=False, num_workers=opt.num_workers)

            self.ext_src_mu, self.ext_src_cov, self.mu_src_ext, self.cov_src_ext, = offline(self.source_loader, self.net, self.opt.n_classes)
            self.bias = self.cov_src_ext.max().item() / 30.
            self.template_ext_cov = torch.eye(self.feat_dim).cuda() * self.bias

            self.ext_src_mu = torch.stack(self.ext_src_mu)
            self.ext_src_cov = torch.stack(self.ext_src_cov) + self.template_ext_cov[None, :, :]
            self.ema_n = torch.zeros(self.opt.n_classes).cuda()
            self.ema_ext_mu = self.ext_src_mu.clone()
            self.ema_ext_cov = self.ext_src_cov.clone()

            self.ema_ext_total_mu = torch.zeros(self.feat_dim).float()
            self.ema_ext_total_cov = torch.zeros(self.feat_dim, self.feat_dim).float()

            self.ema_total_n = 0.

        # Set ema model for knowledge distillation
        self.calibrate_bn_stats(self.opt.bn_epochs)
        self.ema_net = ModelEMA(self.net, decay=opt.ema_momentum).cuda()

        # Parallel Processing
        if opt.dist:
            self.net = sync_batchnorms(self.net)
            self.ema_net = sync_batchnorms(self.ema_net)
            self.net = model_to_DDP(self.net, opt.gpu)

        # dictionary for nearest neighbours
        self.feats_nn_queue = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x feat_dim
        self.prob_nn_queue  = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x n_classes

        # Set augmentation module
        self.normalize_fn = normalize_fn(self.target_loader.dataset.mean, self.target_loader.dataset.std)
        self.denormalize_fn = denormalize_fn(self.target_loader.dataset.mean, self.target_loader.dataset.std)
        self.hard_opt_aug = OptAug(self.ema_net, opt.sub_policy_dim, opt.aug_mult, "Hard",
                                    self.normalize_fn, self.denormalize_fn, self.opt.lmb_norm).cuda()

        # Parallel processing
        if opt.dist:
            self.hard_opt_aug = model_to_DDP(self.hard_opt_aug, opt.gpu).module

        # losses
        self.crterian_cm       = EntropyClassMarginals()

        # optimizer
        if self.opt.dataset_name == "visda" or "imagenet" in self.opt.dataset_name:
            self.optimizer_net = torch.optim.SGD(self.net.parameters(), momentum=0.9, lr=self.opt.lr, weight_decay=self.opt.wd)
        else:
            self.optimizer_net = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)

        # track metrics
        self.metadata_tracker = MetadataTracker()
        self.metric_tracker = MetricTracker()


    @torch.no_grad()
    def update_nearest_neighbours(self, feats, labels):
        hard_labels = torch.argmax(labels, dim=-1)

        for l in range(self.opt.n_classes):
            mask = (hard_labels == l)
            if mask.sum() != 0:
                curr_norm_feats =  feats[mask][:self.opt.nn_queue_size]
                curr_labels = labels[mask][:self.opt.nn_queue_size]

                if self.feats_nn_queue[l].size(0) >= self.opt.nn_queue_size:
                    self.feats_nn_queue[l][:-curr_norm_feats.size(0)] = self.feats_nn_queue[l][curr_norm_feats.size(0):].clone()
                    self.prob_nn_queue[l][:-curr_norm_feats.size(0)] = self.prob_nn_queue[l][curr_norm_feats.size(0):].clone()
                    self.feats_nn_queue[l][-curr_norm_feats.size(0):] = curr_norm_feats.clone()
                    self.prob_nn_queue[l][-curr_norm_feats.size(0):] = curr_labels.clone()
                else:
                    self.feats_nn_queue[l] = torch.cat([self.feats_nn_queue[l], curr_norm_feats], dim=0)
                    self.prob_nn_queue[l] = torch.cat([self.prob_nn_queue[l], curr_labels], dim=0)


    @torch.no_grad()
    def get_pseudo_labels_nearest_neighbours(self, feats):
        norm_feats = F.normalize(feats, dim=-1)
        all_feats_nn_queue = torch.cat(list(self.feats_nn_queue.values()))
        all_feats_nn_queue = F.normalize(all_feats_nn_queue, dim=-1)

        all_prob_nn_queue = torch.cat(list(self.prob_nn_queue.values()))


        # find top-k
        cosine_sim = torch.einsum('ab,cb->ac', norm_feats, all_feats_nn_queue)
        _, idx_neighbours = torch.topk(cosine_sim, k=self.opt.n_neigh, dim=-1) # batch x k

        # get predictions from top-k
        pred_top_k = all_prob_nn_queue[idx_neighbours]

        # get soft voting
        soft_voting = torch.mean(pred_top_k, dim=1)
        pseudo_label = torch.argmax(soft_voting, dim=-1)

        return pseudo_label, soft_voting


    @torch.no_grad()
    def calibrate_bn_stats(self, bn_epochs):
        n_epochs = 0
        print(f"Calibrate BN Stats for {bn_epochs} epochs!")

        # BN calibraton
        while(n_epochs < bn_epochs):
            if self.opt.rank==0:
                pbar = tqdm(total=len(self.target_loader))

            for x, _, _ in self.target_loader:
                x = x.cuda()
                self.net(x[:,0])

                if self.opt.rank==0:
                    pbar.update(1)
            n_epochs += 1

            if self.opt.rank==0:
                pbar.close()


    def set_parameters(self, net):
        for name, param in net.named_parameters():
            if "encoder" not in name and "fc" in name:
                param.requires_grad=False
            else:
                param.requires_grad=True


    # initial evaluation
    @torch.no_grad()
    def evaluate(self, net, data_loader, epoch, title_label="Target"):

        self.metadata_tracker.reset()
        net.eval()

        for x, label, _ in data_loader:

            logit, feat = net(x[:,0].to("cuda", non_blocking=True), True)
            score = self.softmax(logit)

            # Accumulate predictions
            self.metadata_tracker.update_metadata({
                        "Label":label.cpu(),
                        "Pred":torch.argmax(score, dim=-1).cpu(),
                        "Logit_w": score.cpu(),
                        "Feat_w":feat.cpu(),
                    })

        # Aggregate
        self.metadata_tracker.aggregate()

        # Log results
        acc = self.metadata_tracker["Label"]==self.metadata_tracker["Pred"]
        acc_mean = 100 * acc.to(torch.float32).mean()

        # Save predictions to file
        if self.opt.rank==0:
            ensure_dir(os.path.join(self.opt.experiment_dir, f"{title_label}_Evaluation_{epoch}"))
            self.metadata_tracker.to_csv(["Label", "Pred"], os.path.join(self.opt.experiment_dir, f"{title_label}_Evaluation_{epoch}"))

            if epoch == self.opt.n_epochs:
                self.metadata_tracker.to_pkl(["Feat_w", "Logit_w"],  os.path.join(self.opt.experiment_dir, f"{title_label}_Evaluation_{epoch}"))

        net.train()

        return acc_mean


    def cross_entropy_loss(self, p, q):
        return -torch.sum(p * torch.log(q), dim=-1)


    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.opt.lr * decay
        return optimizer


    def train(self):
        """
        Code for training the test time policy networks on the target site.
        """
        print("Training Test Time Policy Network")

        n_iters = 1
        n_epochs = 1
        max_iter = len(self.target_loader)*self.opt.n_epochs

        while n_epochs <= self.opt.n_epochs:
            # Prepare Epoch
            self.metadata_tracker.reset()
            self.metric_tracker.reset()

            if self.opt.dist:
                self.target_loader.sampler.set_epoch(n_epochs)

            if self.opt.rank == 0:
                pbar = tqdm(total = len(self.target_loader))

            # Reset policy predictor and PLR module
            self.feats_nn_queue = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x feat_dim
            self.prob_nn_queue  = {k:torch.Tensor().cuda() for k in range(self.opt.n_classes)} # [nclasses] 256 x n_classes
            self.hard_opt_aug.policy_predictor.reset_weights()

            # Run Epoch
            for i, (x, label, _) in enumerate(self.target_loader):
                if self.opt.apply_lr_scheduler:
                    self.lr_scheduler(self.optimizer_net, n_iters, max_iter)


                ## put current queue and labels on gpu
                batch_x = x[:,random.randint(0, x.size(1)-1)].cuda()
                batch_label = label # used only for computing accuracy

                # Optimize adversarial augmentations
                if not self.opt.no_kl_hard:
                    if self.opt.hard_augment == "optimal":
                        policy_loss = self.hard_opt_aug.optimize(batch_x)

                        self.metric_tracker.update_metrics({
                                "Loss_Policy": policy_loss["loss_policy"].item()
                            },
                            batch_size=batch_x.size(0), compute_avg=True
                        )

                        # sample
                        x_aug_hard = self.hard_opt_aug.sample(batch_x)
                    else:
                        x_aug_hard = self.normalize_fn(self.target_dataset.hard_augment(self.denormalize_fn(batch_x))).unsqueeze(1)

                    x_aug_hard = x_aug_hard.detach() # no gradients

                    if self.opt.debug and n_iters % 5 == 0 and self.opt.rank==0:
                        ensure_dir(os.path.join(self.opt.experiment_dir, f"Evaluation_{n_iters}", "Visuals"))
                        tvu.save_image(self.denormalize_fn(x_aug_hard.flatten(0, 1).detach().cpu()), os.path.join(self.opt.experiment_dir, f"Evaluation_{n_iters}", "Visuals", f"Hard_Aug_imgs.png"), padding=0)
                        tvu.save_image(self.denormalize_fn(x.flatten(0,1).detach().cpu()), os.path.join(self.opt.experiment_dir, f"Evaluation_{n_iters}", "Visuals", f"Easy_Aug_imgs.png"), padding=0)
                        tvu.save_image(self.denormalize_fn(batch_x).cpu(), os.path.join(self.opt.experiment_dir, f"Evaluation_{n_iters}", "Visuals", f"Org_imgs.png"), padding=0)


                # get pseudo labels using the easy augmentations w.r.t. ema model
                with torch.no_grad():
                    scores_ema_easy = torch.Tensor().cuda()
                    feats_ema_easy = torch.Tensor().cuda()

                    for i in reversed(range(x.size(1))):
                        logit, feat = self.ema_net(x[:,i].cuda(), True)
                        score = self.softmax(logit)

                        # scores_ema.append(score)
                        scores_ema_easy = torch.cat([scores_ema_easy, score.unsqueeze(0)])
                        feats_ema_easy = torch.cat([feats_ema_easy, feat.unsqueeze(0)])

                    feats_ema_easy = torch.mean(feats_ema_easy, dim=0)
                    soft_pseudo_labels = torch.mean(scores_ema_easy, dim=0)

                    if self.opt.nn_queue_size > 0:
                        # update the nearest neighbours queue. each gpu has its own queue so gather all feats from all gpus.
                        self.update_nearest_neighbours(concat_all_gather(feats_ema_easy), concat_all_gather(soft_pseudo_labels))

                        # # get nearest neighbour based soft voting
                        _, soft_pseudo_labels = self.get_pseudo_labels_nearest_neighbours(feats_ema_easy)


                # make saved gradients zero
                self.optimizer_net.zero_grad(set_to_none=True)

                # marke marginal distribution uniform by maximizing entropy of class marginals
                # as batch is distributed over gpus, collect them and pass correct gradients
                scores_normal, feats_normal = self.net(batch_x, True)
                scores_normal = self.softmax(scores_normal)

                # compute cross entropy loss between student and teacher
                if self.opt.pl_ce:
                    loss_teach_normal = self.cross_entropy_loss(soft_pseudo_labels, scores_normal).mean()
                    loss_cm_cent = torch.tensor([0.]).cuda()
                elif self.opt.pl_fce:
                    loss_teach_normal = self.cross_entropy_loss(scores_normal, soft_pseudo_labels).mean()
                    loss_cm_cent = torch.tensor([0.]).cuda()
                else:
                    loss_cm_cent  = self.crterian_cm(concat_all_gather(scores_normal, rank=self.opt.rank))*self.opt.world_size
                    loss_teach_normal = self.cross_entropy_loss(scores_normal, soft_pseudo_labels).mean()
                loss = loss_cm_cent.item() + loss_teach_normal.item()

                if self.opt.use_source_stats:
                    b = feats_normal.shape[0]
                    self.ema_total_n += b
                    alpha = 1. / 1280 if self.ema_total_n > 1280 else 1. / self.ema_total_n
                    delta_pre = (feats_normal - self.ema_ext_total_mu.cuda())
                    delta = alpha * delta_pre.sum(dim=0)
                    tmp_mu = self.ema_ext_total_mu.cuda() + delta
                    tmp_cov = self.ema_ext_total_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * self.ema_ext_total_cov.cuda()) - delta[:, None] @ delta[None, :]
                    with torch.no_grad():
                        self.ema_ext_total_mu = tmp_mu.detach().cpu()
                        self.ema_ext_total_cov = tmp_cov.detach().cpu()

                    source_domain = torch.distributions.MultivariateNormal(self.mu_src_ext, self.cov_src_ext + self.template_ext_cov)
                    target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + self.template_ext_cov)
                    loss_stats = (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * 0.05
                    loss += loss_stats.item()

                    self.metric_tracker.update_metrics({
                                "Loss_Stats": loss_stats.item(),
                            },
                            batch_size=batch_x.size(0), compute_avg=True
                        )

                    (loss_cm_cent + loss_teach_normal + loss_stats).backward()
                else:
                    (loss_cm_cent + loss_teach_normal).backward()

                # teacher student knowledge distillation
                feats_hard = None
                if not self.opt.no_kl_hard:
                    for j in range(self.opt.aug_mult):
                        logits_hard, feats_hard = self.net(x_aug_hard[:, j], True)
                        scores_hard = self.softmax(logits_hard)
                        loss_teach = F.kl_div(torch.log(scores_hard), soft_pseudo_labels, reduction="batchmean").mean()*self.opt.lmb_kl
                        loss_teach.backward()
                        loss += loss_teach.item()

                        self.metric_tracker.update_metrics({
                                "Loss_Teach": loss_teach.item(),
                            },
                            batch_size=batch_x.size(0), compute_avg=True
                        )

                self.optimizer_net.step()

                # log Metrics
                self.metric_tracker.update_metrics({
                        "Loss": loss,
                        "Loss_Div": loss_cm_cent.item(),
                        "Loss_Teach_N": loss_teach_normal.item(),
                    },
                    batch_size=batch_x.size(0), compute_avg=True
                )

                log = "Epoch {:d}: Loss[{:.3f}]-Teach_N[{:.3f}]-Teach[{:.3f}]-Div[{:.3f}]-Policy[{:.3f}]".format(
                    n_epochs, self.metric_tracker["Loss"], self.metric_tracker["Loss_Teach_N"], 
                    self.metric_tracker["Loss_Teach"], self.metric_tracker["Loss_Div"], 
                    self.metric_tracker["Loss_Policy"])

                # update teacher
                if self.opt.dist:
                    self.ema_net.update(self.net.module)
                else:
                    self.ema_net.update(self.net)

                # compute accuracy on the current batch
                if self.opt.n_epochs == 1:
                    with torch.no_grad():

                        # student Pred
                        self.net.eval()
                        student_pred, feats_pred = self.net(x[:,0].cuda(), True)
                        student_pred = self.softmax(student_pred)
                        self.net.train()

                        # teacher pred
                        self.ema_net.eval()
                        teacher_pred = self.softmax(self.ema_net(x[:,0].cuda()))
                        self.ema_net.train()

                        self.metadata_tracker.update_metadata({
                            "Label":batch_label.cpu(),
                            "Pred": torch.argmax(teacher_pred, dim= -1).cpu(),
                            "Pred_student": torch.argmax(student_pred, dim= -1).cpu(),
                            "Pred_PLR":torch.argmax(soft_pseudo_labels, dim= -1).cpu(),
                            "Feat_w":feats_pred.cpu(),
                            "Timestamp":torch.ones(x.size(0))*i
                        })

                        if feats_hard is not None:
                            self.metadata_tracker.update_metadata({
                                "Feat_h":feats_hard.cpu()
                            })

                        log += "-Acc_PLR[{:.2f}]".format(100*(self.metadata_tracker["Pred_PLR"]==self.metadata_tracker["Label"]).to(torch.float32).mean())


                else:
                    self.metadata_tracker.update_metadata({
                        "Label":batch_label.cpu(),
                        "Pred": torch.argmax(soft_pseudo_labels, dim= -1).cpu(),
                        "Pred_student": torch.argmax(scores_normal, dim= -1).cpu(),
                        "Feat_w":feats_normal.cpu(),
                        "Timestamp":torch.ones(x.size(0))*i
                    })

                    if feats_hard is not None:
                        self.metadata_tracker.update_metadata({
                            "Feat_h":feats_hard.cpu()
                        })


                log += "-Acc_S[{:.2f}]".format(100*(self.metadata_tracker["Pred_student"]==self.metadata_tracker["Label"]).to(torch.float32).mean())
                log += "-Acc_T[{:.2f}]".format(100*(self.metadata_tracker["Pred"]==self.metadata_tracker["Label"]).to(torch.float32).mean())

                n_iters += 1

                if self.opt.rank == 0:
                    pbar.set_description(log)
                    pbar.update()

            # Terminate epoch
            if self.opt.rank == 0:
                pbar.close()

            print("Loss: {:.3f}".format(self.metric_tracker["Loss"]))

            # Online Evaluation
            if self.opt.n_epochs == 1:
                # Aggregate prediction lists
                self.metadata_tracker.aggregate()

                if self.opt.rank==0:
                    online_acc_teacher = self.metadata_tracker["Label"]==self.metadata_tracker["Pred"]
                    online_acc_teacher = 100 * online_acc_teacher.to(torch.float32).mean()
                    online_acc_student = self.metadata_tracker["Label"]==self.metadata_tracker["Pred_student"]
                    online_acc_student = 100 * online_acc_student.to(torch.float32).mean()
                    online_acc_plr = self.metadata_tracker["Label"]==self.metadata_tracker["Pred_PLR"]
                    online_acc_plr = 100 * online_acc_plr.to(torch.float32).mean()

                    print(f"Online Teacher accuracy/error : %4.4f/%4.4f" % (online_acc_teacher, 100-online_acc_teacher))
                    print(f"Online Students accuracy/error : %4.4f/%4.4f" % (online_acc_student, 100-online_acc_student))
                    print(f"Online PLR accuracy/error : %4.4f/%4.4f" % (online_acc_plr, 100-online_acc_plr))

                    # Save Online predictions to file
                    ensure_dir(os.path.join(self.opt.experiment_dir, f"Online_Evaluation_{n_epochs}"))
                    self.metadata_tracker.to_csv(["Label", "Pred", "Pred_student", "Pred_PLR","Timestamp"], 
                                                    os.path.join(self.opt.experiment_dir, f"Online_Evaluation_{n_epochs}"))
                    self.metadata_tracker.to_pkl(["Feat_w", "Feat_h"], os.path.join(self.opt.experiment_dir, f"Online_Evaluation_{n_epochs}"))

            else:
                # Offline Evaluation
                acc = self.evaluate(self.ema_net, self.eval_target_loader, n_epochs)
                print(f"Teacher accuracy/error : %4.4f/%4.4f" % (acc, 100-acc))
                acc = self.evaluate(self.net, self.eval_target_loader, n_epochs, title_label="Student")
                print(f"Student accuracy/error : %4.4f/%4.4f\n" % (acc, 100-acc))

            if (self.opt.save_every > 0 and n_epochs % self.opt.save_every == 0) or n_epochs==self.opt.n_epochs:
                # Save the network
                torch.save(self.net.state_dict(), os.path.join(self.opt.experiment_dir, f"SLAug_Net_Epoch[{n_epochs}].pth"))

                # Save the augmentation module
                torch.save(self.hard_opt_aug.state_dict(), os.path.join(self.opt.experiment_dir, f"SLAug_Aug_Epoch[{n_epochs}].pth"))

            # Next epoch
            n_epochs +=1
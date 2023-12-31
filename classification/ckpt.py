import torch

# Load the checkpoint
checkpoint = torch.load('/home/uqzxwang/checkpoint/TTA/source/10cls/cifar100_acc99.29_vit_base_patch16_224.pth')

# Remove the "model." prefix from parameter names
new_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('model.'):
        new_key = key[len('model.'):]
        new_state_dict[new_key] = value
    elif key.startswith('normalize.'):
        pass
    else:
        new_state_dict[key] = value

# Update the checkpoint with the modified state dictionary
checkpoint['state_dict'] = new_state_dict

# Save the modified checkpoint
torch.save(checkpoint, '/home/uqzxwang/checkpoint/TTA/source/modified_cifar100_acc99.29_vit_base_patch16_224.pth')
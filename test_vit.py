import timm

available_models = timm.list_models(pretrained=True)

print(available_models)
import torchvision.models as models
print([m for m in dir(models) if 'vit' in m.lower()])

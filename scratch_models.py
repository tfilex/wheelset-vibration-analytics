import torchvision.models as models

archs = [
    ("convnext_tiny", models.convnext_tiny),
    ("efficientnet_v2_s", models.efficientnet_v2_s),
    ("regnet_y_400mf", models.regnet_y_400mf),
    ("swin_t", models.swin_t),
]

for name, builder in archs:
    m = builder()
    print(f"=== {name} ===")
    for n, p in list(m.named_modules())[:15]:
        if isinstance(p, __import__('torch').nn.Conv2d):
            print(f"FIRST CONV: {n} -> {p}")
            break
            
    for n, p in reversed(list(m.named_modules())):
        if isinstance(p, __import__('torch').nn.Linear):
            print(f"FINAL LINEAR: {n} -> {p}")
            break
    print()
    
print("Mobile models:")
print([m for m in dir(models) if 'mobile' in m.lower()])

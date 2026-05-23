import torchvision.models as models

m = models.maxvit_t()
print("=== maxvit_t ===")
for n, p in list(m.named_modules())[:15]:
    if isinstance(p, __import__('torch').nn.Conv2d):
        print(f"FIRST CONV: {n} -> {p}")
        break

for n, p in reversed(list(m.named_modules())):
    if isinstance(p, __import__('torch').nn.Linear):
        print(f"FINAL LINEAR: {n} -> {p}")
        break

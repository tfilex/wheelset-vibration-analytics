import torch
import torchvision.models as models

m2 = models.maxvit_t(weights=None)
m2.stem[0][0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
m2.classifier[5] = torch.nn.Linear(512, 10, bias=False)

dummy = torch.randn(2, 1, 224, 224)
try:
    out2 = m2(dummy)
    print("MaxVit 224 OK:", out2.shape)
except Exception as e:
    print("MaxVit ERROR:", e)

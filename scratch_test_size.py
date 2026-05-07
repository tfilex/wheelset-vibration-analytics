import torch
import torchvision.models as models

m = models.swin_t(weights=None)
m.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
m.head = torch.nn.Linear(768, 10)

dummy = torch.randn(2, 1, 129, 64)
try:
    out = m(dummy)
    print("Swin OK:", out.shape)
except Exception as e:
    print("Swin ERROR:", e)

m2 = models.maxvit_t(weights=None)
m2.stem[0][0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
m2.classifier[5] = torch.nn.Linear(512, 10, bias=False)

try:
    out2 = m2(dummy)
    print("MaxVit OK:", out2.shape)
except Exception as e:
    print("MaxVit ERROR:", e)

m3 = models.convnext_tiny(weights=None)
m3.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
m3.classifier[2] = torch.nn.Linear(768, 10)
try:
    out3 = m3(dummy)
    print("ConvNeXt OK:", out3.shape)
except Exception as e:
    print("ConvNeXt ERROR:", e)

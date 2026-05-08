import torch
import torchaudio.transforms as T

dummy = torch.randn(2, 1, 129, 9)
time_mask = T.TimeMasking(time_mask_param=15)
freq_mask = T.FrequencyMasking(freq_mask_param=15)

try:
    out = time_mask(dummy)
    out = freq_mask(out)
    print("Success! Shape:", out.shape)
except Exception as e:
    print("Error:", e)

import torch
from medseglab.models.unetr import UNETRLightning

def test_forward_cpu_smoke():
    model = UNETRLightning(in_channels=1, out_channels=1, img_size=(32,32,32), feature_size=16)
    x = torch.randn(1,1,32,32,32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,1,32,32,32)

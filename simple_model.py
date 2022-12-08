import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from PIL import Image

resnet = resnet18(pretrained=True)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.register_parameter(name='weights', param=nn.Parameter(torch.randn(1,256,56,56)))
        self.blocks = nn.Sequential(
            # resnet.conv1,
            nn.Conv2d(3, 64, 7, bias=False, padding=3, stride=2),
            # nn.Linear(2, 4, bias=False),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2),
            # nn.InstanceNorm2d(4),
            # nn.Sigmoid(),
            # nn.Softmax(-1),
        )
        self.resnet = resnet18(pretrained=True)
        # self.blocks[0].weight.data.fill_(1.2)

    def forward(self, x):
        x = x[:,:,:,:3]
        x = x.permute(0,3,1,2)

        x = x / (255. * torch.tensor([0.229, 0.224, 0.225])[None,:,None,None]) - \
                        torch.tensor([0.485, 0.456, 0.406])[None,:,None,None]
        return self.resnet(x)
        # return x.permute(1, 0)

model = Model()
model = model.eval()
# model = resnet # Model()
# x = torch.arange(0,16, dtype=torch.float32).view(4,4)
#x = torch.ones(2,2)
# x = torch.ones(1,4,64,64)
x = torch.ones(1,224,224,4)
#x = torch.arange(4, dtype=torch.float32).view(1,1,1,4)
# x = torch.ones(1,4,64,64)
# x = torch.ones(2)

#with torch.no_grad():
    #print(model(x))

torch.onnx.export(
    model,
    x,
    "./wnn-wasm/resnet18.onnx",
    do_constant_folding=True,
)

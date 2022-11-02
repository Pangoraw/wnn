import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

resnet = resnet50(pretrained=True)

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
        # self.blocks[0].weight.data.fill_(1.2)

    def forward(self, x):
        return self.blocks(x).int()
        # return x.permute(1, 0)

model = Model()
# model = resnet # Model()
# x = torch.arange(0,16, dtype=torch.float32).view(4,4)
x = torch.ones(1,3,224,224)
# x = torch.ones(1,4,64,64)

with torch.no_grad():
    print(model(x))

torch.onnx.export(
    model,
    x,
    "./simple_model.onnx",
    do_constant_folding=True,
)

import torch
from torch import nn
from torchvision.models import resnet50

resnet = resnet50(pretrained=True)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter(name='weights', param=nn.Parameter(torch.randn(1,256,56,56)))
        self.blocks = nn.Sequential(
            # resnet.conv1,
            # nn.Conv2d(3, 64, 7, bias=False, padding=3, stride=2),
            # nn.Linear(2, 4, bias=False),
            # nn.ReLU(),
        )
        # self.blocks[0].weight.data.fill_(1.2)

    def forward(self, x):
        # return self.blocks(x)
        return x + self.weights


# model = Model()
model = resnet # Model()
x = torch.ones(1,3,224,224)
# x = torch.ones(1,256,56,56)

with torch.no_grad():
    print(model(x))

torch.onnx.export(
    model,
    x,
    "./simple_model.onnx",
    do_constant_folding=True,
)


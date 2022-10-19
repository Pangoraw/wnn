import torch
from torch import nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 6, 3, bias=False),
            # nn.Linear(2, 4, bias=False),
            # nn.ReLU(),
        )
        self.blocks[0].weight.data.fill_(1.2)

    def forward(self, x):
        return self.blocks(x)


model = Model()
x = torch.ones(1,3,10,10)

with torch.no_grad():
    print(model(x))

torch.onnx.export(
    model,
    x,
    "./simple_model.onnx",
    do_constant_folding=True,
)


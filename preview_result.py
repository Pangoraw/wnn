import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

out_folder = "activations/" 
out = out_folder + os.listdir(out_folder)[0]
x = torch.tensor(np.load(out))
image = x.squeeze().permute(1, 2, 0)
image = (image / 2 + .5).clamp(0, 1).float()
image = (image * 255).numpy().astype(np.uint8)
img = Image.fromarray(image)

plt.imshow(img)
plt.axis("off")
plt.show()

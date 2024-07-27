import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import os

for gender in ['f', 'm']:
    for i in range(10):
        img = Image.open(f'./data/physical_images/正文展示/{gender}{i}.png').convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Resize((2500, 2500))(img)
        saved_path = './data/physical_imgs'
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        vutils.save_image(img, f'./data/physical_imgs/{gender}{i}.jpg')

import torch
from torchvision import datasets, transforms
from skimage import color
import os
from PIL import Image
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Define Batch size
batch_size = 32

# Define the percentage of data to be used for validation
validation_split = 0.2


class LABtoRGBDataset(Dataset):
    def __init__(self, root, rgb_transform=None,lab_transform=None):

        self.root = root
        self.rgb_transform = rgb_transform
        self.lab_transform = lab_transform
        self.pil_to_tensor = transforms.ToTensor()
        self.image_list = [img_path for img_path in glob.glob(f"{self.root}/*/*/*") if (img_path.endswith('.jpg') | img_path.endswith('.jpeg'))]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_path = self.image_list[idx]

        rgb_image = Image.open(image_path).convert('RGB')

        lab_image , rgb_image_transformed = self.preprocess_rgb_to_lab(rgb_image)

        return {"input_image" : lab_image,
              "output_image" : self.pil_to_tensor(rgb_image_transformed)}

    def preprocess_rgb_to_lab(self,rgb_image) :

      # Transformations on rgb image
      if self.rgb_transform :
        rgb_image_transformed = self.rgb_transform(rgb_image)

      # Convert RGB to LAB
      lab_image = color.rgb2lab(rgb_image_transformed)

      # Rescaling
      lab_image = (lab_image + [0, 128, 128]) / [100, 255, 255]  # Normalize L*, a*, b*

      if self.lab_transform:
          lab_image = self.lab_transform(lab_image)

      return lab_image ,rgb_image_transformed

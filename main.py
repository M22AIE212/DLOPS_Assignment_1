
## Setting up Tensorboard
from torch.utils.tensorboard import SummaryWriter

## Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import itertools

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import cv2
from torch.utils import data
from functools import reduce
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

from LABtoRGBDataset import LABtoRGBDataset
from RGBtoHorFlipDataset import RGBtoHorFlipDataset
from RGBtoNegDataset import RGBtoNegDataset

from autoencoder import Autoencoder
from train_one_epoch import train_one_epoch
from reconstruction_metrics import calculate_METRICS
import argparse


if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description='Process some arguments for the script.')

  parser.add_argument('--device', type=str, default='cuda', help='Device for computation (e.g., cuda, cpu)')
  parser.add_argument('--loss_fn', type=str, default='mse', help='Loss function to be used ("mse" or "mae" )')
  parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer name (e.g., adam, sgd)')
  parser.add_argument('--epoch_number', type=int, default=0, help='Current epoch number')
  parser.add_argument('--EPOCHS', type=int, default=1, help='Total number of epochs')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--b_dim', type=int, default=400, help='Latent Dimension for Autoencoder')
  parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
  parser.add_argument('--task', type=str, default='lab_to_rgb', help='Task description, "lab_to_rgb" or "rgb_to_hflip" or "rgb_to_neg"')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
  parser.add_argument('--dir', type=str, default=os.getcwd(), help='Loss Directory path')
  parser.add_argument('--model_dir', type=str, default=os.getcwd(), help='Model directory path')

  args = parser.parse_args()

  device = args.device
  loss_fn = args.loss_fn
  opt_name = args.opt_name
  epoch_number = args.epoch_number
  EPOCHS = args.EPOCHS
  lr = args.lr
  b_dim = args.b_dim
  validation_split = args.validation_split
  task = args.task
  batch_size = args.batch_size
  train_loss_dir_path = args.dir
  val_loss_dir_path = args.dir
  model_dir = args.model_dir

  # default `log_dir` is "runs" - we'll be more specific here
  writer = SummaryWriter(f'/content/drive/MyDrive/Assignments/DLOPS/Assignment1/codes/runs_{task}/autoencoder_experiment')

  ## 1. Model Object Instantiation
  model = Autoencoder(b_dim).to(device)

  ## 2. Loss fn
  if loss_fn == "mse" :
    loss_fn = torch.nn.MSELoss().cuda()
  elif loss_fn == "mae" :
    loss_fn = torch.nn.L1Loss().cuda()

  ## 3. Using an Adam Optimizer
  if opt_name == "adam" :
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  elif opt_name == "sgd" :
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

  ## 4. Create Datasets
  best_vloss = 1_000_000.
  train_loss = []
  val_loss = []

  if task == "lab_to_rgb" :
    rgb_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToPILImage(),
        ])
    lab_transforms = transforms.Compose([
        transforms.ToTensor()
        ])

    # Update the dataset and dataloaders
    dataset = LABtoRGBDataset(root='./hymenoptera_data/hymenoptera_data', rgb_transform=rgb_transforms,lab_transform = lab_transforms)

  elif task == "rgb_to_hflip" :
    rgb_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ,
        transforms.ToPILImage(),
      ])

    # Update the dataset and dataloaders
    dataset = RGBtoHorFlipDataset(root='./hymenoptera_data/hymenoptera_data', rgb_transform=rgb_transforms)
  elif task == "rgb_to_neg" :
    rgb_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ,
        transforms.ToPILImage(),
        ])


    # Update the dataset and dataloaders
    dataset = RGBtoNegDataset(root='./hymenoptera_data/hymenoptera_data', rgb_transform=rgb_transforms)


  ## 5. Create  Data Loaders
  # Calculate the sizes of training and validation sets
  dataset_size = len(dataset)
  val_size = int(validation_split * dataset_size)
  train_size = dataset_size - val_size

  # Splitting the Pytorch Dataset object into train and validation
  # Use random_split to create training and validation subsets
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  # Create DataLoader instances for training and validation
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

  ## 6. Training

  for epoch in tqdm(range(EPOCHS)):
      print('EPOCH {}:'.format(epoch_number + 1))

      # Make sure gradient tracking is on, and do a pass over the data
      model.train(True)
      avg_loss,model = train_one_epoch(epoch_number,train_loader,model,loss_fn,optimizer)

      running_vloss = 0.0
      # Set the model to evaluation mode, disabling dropout and using population
      # statistics for batch normalization.
      model.eval()

      # Disable gradient computation and reduce memory consumption.
      with torch.no_grad():
          torch.manual_seed(2809)
          for i, batch in enumerate(val_loader):
              vinputs, voutput_image =  batch["input_image"].float().cuda() , batch["output_image"].float().cuda()
              voutputs = model(vinputs)
              vloss = loss_fn(voutputs, voutput_image)
              running_vloss += vloss.item()

      avg_vloss = (running_vloss / len(val_loader))
      train_loss.append((epoch,avg_loss))
      val_loss.append((epoch,avg_vloss))
      print(' LOSS train : {} , valid {} , LR : {}'.format(avg_loss, avg_vloss,optimizer.param_groups[0]['lr']))

      # Log training loss to TensorBoard
      writer.add_scalar(f'Training MSE Loss', avg_loss, epoch + 1)

      # Log Validation loss to TensorBoard
      writer.add_scalar(f'Validation MSE Loss', avg_vloss, epoch + 1)

      # Track best performance, and save the model's state
      if avg_vloss < best_vloss:
          best_vloss = avg_vloss
          model_path = f'{model_dir}/task_{task}_best_model_epochs_{EPOCHS}_lr_{lr}_optimizer_{opt_name}_lossfn_{loss_fn}_bottleneck_dim_{b_dim}.pth'
          torch.save(model.state_dict(), model_path)

      epoch_number += 1


  train_loss_df = pd.DataFrame(train_loss,columns = ['epoch','loss'])
  val_loss_df = pd.DataFrame(val_loss,columns = ['epoch','loss'])
  train_loss_df.to_csv(train_loss_dir_path + f"/task_{task}_train_optimizer_{opt_name}_lossfn_{loss_fn}_bottleneck_dim_{b_dim}.csv")
  val_loss_df.to_csv(val_loss_dir_path + f"/task_{task}_val_optimizer_{opt_name}_lossfn_{loss_fn}_bottleneck_dim_{b_dim}.csv")

  print("*"*50)
  print()
  print("Calculating Reconstruction Metrics : ")
  print()
  print("*"*50)
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  reconstruction_loss_mse,reconstruction_loss_mae,average_rmse, average_ssim, average_psnr= calculate_METRICS(model, val_loader,device)

  print(f"MSE on test dataset : {reconstruction_loss_mse}")
  print(f"MAE on test dataset : {reconstruction_loss_mae}")
  print(f"RMSE on test dataset : {average_rmse}")
  print(f"SSIM on test dataset : {average_ssim}")
  print(f"PSNR on test dataset : {average_psnr}")

  print("*"*50)
  print()
  print("Plotting Images : ")
  print()
  print("*"*50)
  print()
  # Get a batch of data and pass it through the autoencoder
  torch.manual_seed(2809)

  for test_batch in val_loader :
    images, output = test_batch['input_image'].float().cuda() , test_batch['output_image'].float().cuda()
  reconstructed_images = model(images)

  # Plot the original and reconstructed images
  fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))
  for i in range(5):
      # Original image
      original_image = images[i].permute(1, 2, 0).cpu().numpy()  # Permute channels to (height, width, channels)
      axes[i, 0].imshow(original_image)
      axes[i, 0].set_title("Input")

      # Modified image
      original_image = output[i].permute(1, 2, 0).cpu().numpy()  # Permute channels to (height, width, channels)
      axes[i, 1].imshow(original_image)
      axes[i, 1].set_title("Groung Truth Output")

      # Reconstructed image
      reconstructed_image = reconstructed_images[i].permute(1, 2, 0).detach().cpu().numpy()
      axes[i, 2].imshow(reconstructed_image)
      axes[i, 2].set_title("Reconstructed Output")

  plt.tight_layout()
  plt.show()

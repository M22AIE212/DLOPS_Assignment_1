import wget
import zipfile
import os
import pandas as pd
from functools import reduce
import torch
import numpy as np
from random import random
import random

def Hymenoptera_Data_Download(dir_path = '/content/hymenoptera_data') :

  url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
  zip_file_path = './hymenoptera_data.zip'
  extract_to = dir_path

  ## Creating Directory
  if ~os.path.exists('./hymenoptera_data') == True :
    os.mkdir(dir_path)

  ## Download Data
  try:
    print("Downloading Data ... ")
    wget.download(url, './')
  except Exception as e:
      print(e)

  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
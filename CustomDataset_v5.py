from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import torch

class CustomDataset(Dataset):
  def __init__(self, image_dirs, image_labels):
      self.image_dirs = image_dirs  # ids list
      self.image_labels = image_labels  # labels list
      self.transform = transforms.Compose([transforms.Resize((300,300)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

  def __len__(self):
    return len(self.image_labels)

  def __getitem__(self, idx):
    image = Image.open(self.image_dirs[idx])
    feature = self.feature_process(self.image_dirs[idx])

    if self.transform:
      image = self.transform(image)
    label = self.image_labels[idx]

    return image, label, feature

  def feature_process(self, dir):
      dir = '.'+(dir.split('.')[1])+'.csv'
      csv = pd.read_csv(dir)
      df = pd.DataFrame(csv.iloc[:, [1,2,3, 25,26,27, 28,29,30]])
      df = df.replace('-', np.nan)
      df = df.fillna(method='ffill')
      df = df.fillna(0)
      df = df.astype({'내부 이슬점 평균': 'float64','내부 이슬점 최저': 'float64','내부 이슬점 최고': 'float64',
                      '내부 CO2 평균': 'float64','내부 CO2 최저': 'float64','내부 CO2 최고': 'float64'})

      row = [df[n].mean() for n in df.columns]

      return torch.Tensor(row)
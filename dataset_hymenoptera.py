from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import numpy

class HYMENOPTERADataset(Dataset):
    def __init__(self, path):
        self.scene = datasets.ImageFolder(root=path, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    ]))

    def __getitem__(self, index):
      if isinstance(index, slice):
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else len(self)
        step = index.step if index.step is not None else 1
        indices = range(start, stop, step)
        data = []
        targets = []
        for i in indices:
            if isinstance(i, numpy.float64):
                i = i.astype(numpy.int64)
            d, t = self.scene[i]
            data.append(d)
            targets.append(t)
        return data, targets, indices
      if isinstance(index, np.ndarray):
        data = []
        targets = []
        for i in index:
            if isinstance(i, numpy.float64):
                i = i.astype(numpy.int64)
            d, t = self.scene[i]
            data.append(d)
            targets.append(t)
        return data, targets

      if isinstance(index, numpy.float64):
        index = index.astype(numpy.int64)
      data, target = self.scene[index]

      return data, target, index

    def __len__(self):
        return len(self.scene)

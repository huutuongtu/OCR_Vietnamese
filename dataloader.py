from cmath import acos
import torch
from torch.utils.data import Dataset
import pandas as pd
from char_embedding import tensor_to_text,text_to_tensor
import numpy as np
import cv2


data = pd.read_csv("/home/tuht/DL/data.csv")
sample = data.shape[0]
cols = ['Path', 'Labels']

class ORC_Dataset(Dataset):

    def __init__(self):
        self.n_samples = sample
        Path = data['Path']
        Labels = data['Labels']
        self.A = Path
        self.B = Labels
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        img = '/home/tuht/DL/vn_handwritten_images/data/' + str(self.A[index])
        # print(self.B[index])
        img = cv2.imread(img)
        img = img/np.max(img)
        img = cv2.resize(img, (100, 100))
        image = cv2.transpose(img)
        image = image.reshape(3, 100, 100)
# Đặt kích thước của ảnh

        image = torch.tensor(image)
        label = text_to_tensor(self.B[index])
        return image, label, self.B[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



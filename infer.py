import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet18 import ResNet18
from torch.utils.data import DataLoader, TensorDataset
from dataloader import ORC_Dataset
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
from jiwer import wer, cer


batch_size = 1

data = ORC_Dataset()
dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True,)

# Define the ResNet-18 model and loss function
device = 'cuda'

#input: batch x channel x dai x rong 
net = ResNet18().to(device)
net = torch.load("./checkpoint/model_checkpoint.pth")
# Train the ResNet-18 model
CER = 0
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, label, ground_truth = data
        inputs, label = inputs.to(torch.float).to(device), label.to(torch.float).to(device)

        # optimizer.zero_grad()

        outputs = net(inputs)
        # print(labels.shape)
        #1024 x 141
        
        # print(outputs.shape) #batch x 256 x numberofclass
        # outputs = (F.log_softmax(outputs, dim=2))
        # print(torch.argmax(outputs, dim=2))
        # print(outputs.shape)
        x = F.log_softmax(outputs,dim=2)
        # x = x.unsqueeze(1)
        labels = ['', 'W', 'ỉ', 'I', 'ở', '/', 'ò', 'E', 'p', 'K', 's', 'â', 'ô', 'd', 'Ứ', '6', 'í', 'ạ', '3', 'V', 'c', '4', "'", 'ỗ', 'h', 'Ô', 'ồ', 'l', ':', 'ổ', 'Ê', 'ặ', 'ữ', 'Đ', 'C', 'á', 'u', 'r', 'ố', 'M', 'w', 'y', 'ụ', '1', 'ứ', 'ệ', 'ị', 'ư', 'o', 'ù', 'ủ', 'O', 'e', 'T', 'Y', 'k', '#', 'A', 'ỵ', '+', 'ý', '8', 'è', 'ĩ', 'ằ', 'ơ', '(', 'ợ', 'ũ', 'G', 'à', 'S', 'L', 'ỳ', '9', 'ă', '0', 'ẽ', 'ừ', '5', 'ấ', 'ó', 'đ', 'U', 'g', 'ọ', 'ờ', 'ỏ', 'H', 'ỡ', 'ã', 'm', 'Â', 'b', 'ế', 'ẻ', 'ầ', 'ề', 'ú', 'z', '2', 'Q', 'ẩ', 'v', 't', 'J', 'ắ', 'q', 'ì', 'X', 'ộ', '7', 'ả', 'ớ', 'D', 'Ơ', 'P', 'B', 'ễ', '.', 'x', 'F', 'ê', ')', 'ử', ',', 'ỷ', 'ể', 'ẵ', 'n', 'i', 'a', 'õ', 'é', 'ậ', 'ự', ' ', 'N', '-', 'R', 'ỹ']
        x = x.squeeze(1)
        x = x.detach().cpu().numpy()
        decoder = build_ctcdecoder(
            labels = labels,
            
        )
        # print(x.shape)
        # x = x.squeeze(0)
        # ground_truth = (label)
        
        # print()
        hypothesis = str(decoder.decode(x))
        CER += cer(str(ground_truth).split("'")[1], hypothesis) 
        # print(hypothesis)
    
# print(CER/1838)
#CER: 1.33%


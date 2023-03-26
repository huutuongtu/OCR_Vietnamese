import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet18 import ResNet18
from torch.utils.data import DataLoader, TensorDataset
from dataloader import ORC_Dataset
import torch.nn.functional as F



batch_size = 1

data = ORC_Dataset()
dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True,)

# Define the ResNet-18 model and loss function
device = 'cuda'

#input: batch x channel x dai x rong 
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank = 0).to('cuda')
optimizer = optim.AdamW(net.parameters(), lr = 0.001)
# Train the ResNet-18 model
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(torch.float).to(device), labels.to(torch.float).to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        # print(labels.shape)
        #1024 x 141
        
        # print(outputs.shape) #batch x 256 x numberofclass
        outputs = (F.log_softmax(outputs, dim=2))
        # print(labels.shape)
        # print(outputs.shape)
        input_lengths = torch.tensor([256]).to('cuda')
        target_lengths =torch.tensor([labels.shape[1]]).to('cuda')
        loss = ctc_loss(outputs, labels, input_lengths, target_lengths)
        # loss = criterion(outputs, labels)
        if i%100==0:
            print(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, loss: {running_loss / len(data)}")

    torch.save(net, '/home/tuht/DL/checkpoint/model_checkpoint.pth')


import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        # print(x.shape) #batch x time x in_features
        x = x.unsqueeze(1)  # add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        # flatten the output of the convolutional layers
        # x = x.view(x.size(0), -1)
        # apply the linear layer
        x = self.linear(x)
        x = x.view(x.size(0), -1, self.linear.out_features)
        # print(x.shape) #batch x time x outfeatures
        return x
    



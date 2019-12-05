import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, 1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(6 * 6 * 512, 1024)
        self.fc2 = nn.Linear(1024, 136)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1)
        pool1 = self.pool1(relu1)

        conv2 = self.conv2(pool1)
        relu2 = F.relu(conv2)
        pool2 = self.pool2(relu2)

        conv3 = self.conv3(pool2)
        relu3 = F.relu(conv3)
        pool3 = self.pool3(relu3)

        conv4 = self.conv4(pool3)
        relu4 = F.relu(conv4)
        pool4 = self.pool4(relu4)

        conv5 = self.conv5(pool4)
        relu5 = F.relu(conv5)
        pool5 = self.pool5(relu5)

        drop_l = self.drop(pool5)
        compress = drop_l.view(drop_l.size(0), -1)
        final_relu = F.relu(self.fc1(compress))
        fc = self.fc2(final_relu)
        return fc

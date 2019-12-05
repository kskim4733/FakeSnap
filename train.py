import torch
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from FaceDataSet import FaceDataSet
from model import Net
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from random import shuffle
#-----------------------------------------------------------------------------------------------------------------------
def start_train(model, epochs, root_path, type_path, fileName):
    train_set = FaceDataSet(root_path, type_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
#Data Loader------------------------------------------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    model.to(device)
# Training Param--------------------------------------------------------------------------------------------------------
    for epoch in range(epochs):
        running_loss = 0.0
        for index, (img, pts) in enumerate(train_loader):
            images = img.float().to(device)
            key_pts = pts.view(pts.size(0), -1).float().to(device)
            prediction = model(images)
            loss = criterion(prediction, key_pts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if index % 10 == 0:  # print every 10 batches
                print('Epoch: {}, AVGLoss: {}'.format(epoch + 1, running_loss / 1000))
                running_loss = 0.0
        if (epoch > epochs - 10):
            save_dir = 'C:\\Users\\kskim\\Desktop\\train-test-data'
            save_path = os.path.join(save_dir, "checkt"+str(epoch)+".pt")
            torch.save(model.state_dict(), save_path)
# Save Model------------------------------------------------------------------------------------------------------------
    save_dir = 'C:\\Users\\kskim\\Desktop\\train-test-data'
    save_path = os.path.join(save_dir, fileName)
    torch.save(model.state_dict(), save_path)


def test_model(model, model_path, img_path):
    # this function uses trained model to get keypoint and viusalize it on top of the test img
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = input_image / 255.0

    input_image_tensor = np.expand_dims(input_image, 0)
    input_image_tensor = np.expand_dims(input_image_tensor, 0)
    input_image_tensor = Variable(torch.from_numpy(input_image_tensor))
    input_image_tensor = input_image_tensor.float().to(device)

    prediction = model(input_image_tensor).cpu().detach()
    plot_points(input_image, prediction)


def plot_points(image, keypoints): # used to plot points on top of the image
    plt.figure()
    keypoints = keypoints.data.numpy()
    keypoints = (keypoints * 50.0) + 100
    keypoints = np.reshape(keypoints, (68, -1))

    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, marker='.', c='r')
    plt.show()

if __name__ == "__main__":
    model = Net()
    train = True    # if set to true, running this code will train the model
    vibration_test = False #  set to true to see the result of key points using only one image
    if (train):
        start_train(model, 50, "C:\\Users\\kskim\\Desktop\\train-test-data", "training", 'test.pt')
    else:
        model_path = "C:\\Users\\kskim\\Desktop\\train-test-data\\test.pt"
        path = "C:\\Users\\kskim\\Desktop\\train-test-data\\test"
        onlyfiles = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
        shuffle(onlyfiles)
        if (vibration_test):
            while(True):
                imgPath = onlyfiles[5]
                test_model(model, model_path, imgPath)
        else:
            for imgPath in (onlyfiles):
                img_path = imgPath
                test_model(model, model_path, img_path)

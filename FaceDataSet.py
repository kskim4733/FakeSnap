import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import torch
import random

class FaceDataSet(Dataset):
    def randomCrop(self, pre_image, pre_pts, crop_size):
        pre_image = np.asarray(pre_image)
        height, width = self.img_y_size, self.img_x_size
        new_h, new_w = (crop_size, crop_size)

        top = np.random.randint(0, height - new_h)
        left = np.random.randint(0, width - new_w)

        image = pre_image[top: top + new_h, left: left + new_w]
        key_pts = pre_pts - [left, top]


        return image, key_pts


    def tensorize(self, pre_image, pre_pts):
        img_tensor = torch.from_numpy(pre_image)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        pts_tensor = torch.from_numpy(pre_pts)

        return img_tensor, pts_tensor


    def readData(self, data):
        # read it as np array using cv2, apply  grayscale, resize, normalize
        img_path = data['img_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        raw_pts = np.asarray(data['pts'])
        x_ratio = float(self.img_x_size / image.shape[1]) # getting scale ratio for x and y
        y_ratio = float(self.img_y_size / image.shape[0])
        scaled_pts = raw_pts * (x_ratio, y_ratio)


        scaled_image = cv2.resize(image, (self.img_x_size, self.img_y_size))
        # scaled_image = cv2.fastNlMeansDenoisingColored(scaled_image, None, 10, 10, 7, 21)
        # scaled_image = cv2.blur(scaled_image, (5, 5))
        normalized_img = scaled_image / 255.0

        return normalized_img, scaled_pts


    def randomHorizontal(self, image, pts):
        if bool(random.getrandbits(1)):
            image = cv2.flip(image, 1)
            for index in range(0, len(pts)):
                pts[index][0] = image.shape[0] - pts[index][0]
        return image, pts

    def __init__(self, root_path, img_path):
        self.img_x_size, self.img_y_size = 250, 250
        self.root_path = root_path
        self.img_path = img_path

        self.img_path = os.path.join(root_path, img_path)
        self.csv_path = os.path.join(root_path, img_path + ".csv")

        print("Image Path: ", self.img_path)
        print("CSV Path: ", self.csv_path)
        print("v4")
        self.images_list = [name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))]
        data_point = pd.read_csv(self.csv_path)
        self.img_pt_list = []
        print("Loading DataSet...")
        for img_name in tqdm(self.images_list):
            file_index = (data_point.loc[data_point['img_name'] == img_name].index[0])
            x_list = (data_point.iloc[file_index, 1::2])
            y_list = (data_point.iloc[file_index, 2::2])

            img_path = os.path.join(self.img_path, img_name)
            pts = list(zip(x_list, y_list))
            self.img_pt_list.append({"img_path": img_path, "pts": pts})


    def __getitem__(self, index):
        retrieved = self.img_pt_list[index]
        image, pts = self.readData(retrieved)  # apply resize, normalize, grayscale
        image, pts = self.randomCrop(image, pts, 224)  # apply random crop
        # image, pts = self.randomHorizontal(image, pts)  # commented out for testing purposes
        pts = (pts - 100) / 50  # normalize key point
        img_tensor, pts_tensor = self.tensorize(image, pts)
        return img_tensor, pts_tensor
        # return image, pts


    def __len__(self):
        return len(self.images_list)


def view_point(image, pts):
    x_list, y_list = zip(*pts)
    plt.scatter(x=x_list, y=y_list, c='r', s=10)
    plt.imshow(image, cmap='gray')
    plt.show()



if __name__ == "__main__":
    data_root_path = "C:\\Users\\kskim\\Desktop\\train-test-data"
    data_test_path = "test"
    data_training_path = "training"

    data_set = FaceDataSet(data_root_path, data_training_path) # length 3462
    for i in range(0, len(data_set)):
        img, pts = data_set[i]
        print(img.shape, pts.shape)
        view_point(img, pts)

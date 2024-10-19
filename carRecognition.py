import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
torch.manual_seed(2)

NEW_X = 338
NEW_Y = 190

# Create the model class using sigmoid as the activation function

# class CarRecognitionNet(nn.Module):
#     def __init__(self, in_size):
#         super(CarRecognitionNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         self.fc1_input_size = (NEW_X // 4) * (NEW_Y // 4) * 32
#         self.fc1 = nn.Linear(self.fc1_input_size, 128)  # Modifiez selon la taille de votre image
#         self.fc2 = nn.Linear(128, 4)  # 4 pour les coordonnées (x_min, y_min, x_max, y_max)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)

#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
        
#         x = x.view(x.size(0), -1)  # Aplatir // x.size(0) pour garder le batch size flexible
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)  # Prédictions des coordonnées
#         return x
    

# train function, x.shape  = torch.Size([32, 3, 128, 128])
# x.size CRN forward : [32, 49152] 
# mat1 and mat2 shapes cannot be multiplied (49152x32 and 49152x128)


# Train the model

def train(model, criterion, train_loader, test_loader, optimizer, epochs=10):
    i = 0
    output = {'training_loss': [], 'test_accuracy': []}  
    
    for epoch in range(epochs):
        for i, (image, box) in enumerate(train_loader):
            image=image.float()
            box=box.float()
            optimizer.zero_grad()
            z = model(image.unsqueeze(0)) #x.view(-1, NEW_X * NEW_Y * 3)
            loss = criterion(z, box)
            loss.backward()
            optimizer.step()
            output['training_loss'].append(loss.data.item())
        
        # correct = 0
        # for x, y in test_loader:
        #     z = model(x.view(-1, 128 * 128 * 3))
        #     _, label = torch.max(z, 1)
        #     print(z)
        #     correct += (label == y).sum().item()
    
        # accuracy = 100 * (correct / len(test_loader))
        # output['test_accuracy'].append(accuracy)
    
    return output



class CarRecognitionDataset(Dataset):
    def __init__(self, images_dir, csv_annotations, transform=None):
        self.images_dir = images_dir
        self.annotations = pd.read_csv(csv_annotations)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = image[:,:,:3] # removing alpha cannal if it exists
        image_resized = cv2.resize(image, (NEW_X, NEW_Y))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() # readapte les canaux par permute
        
        y = torch.tensor(self.annotations.iloc[index, 1:5]) # 1:5 for bounding box coord
        # y = y/2 # change to ratio

        if self.transform:
            image_transformed = self.transform(image_resized)
            return image_transformed, y

        return image_tensor, y

        
        

def display_bounding_boxes(image, pred):
    """
    Affiche l'image avec les bounding boxes.

    Parameters:
    - image: l'image à afficher (par exemple sous forme de tableau NumPy)
    - pred: liste des bounding boxes prédictes, chacune étant sous la forme (x_min, y_min, x_max, y_max)

    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in [pred.squeeze(0).tolist()]:
        x_min, y_min, x_max, y_max = box
        
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

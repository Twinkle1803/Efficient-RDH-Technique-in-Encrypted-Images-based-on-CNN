import os
import argparse
import torchvision
from torch.utils.data import DataLoader
from Dataset_new import MyData
import torch
from torch import nn
import numpy as np
import cv2

# Check if MPS (Apple Silicon) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Argument parser to take the train state as input
parser = argparse.ArgumentParser(description="Run ACCN_test_mps.py with a specified train state.")
parser.add_argument("train_state", type=int, help="Train state number (e.g., 300)")
args = parser.parse_args()

train_state = args.train_state

############## Huffman Encoding Implementation #################
## Node class
class Node(object):
    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None

## Huffman Tree class
class HuffmanTree(object):
    def __init__(self, char_weights):
        self.a = [Node(part[0], part[1]) for part in char_weights]
        while len(self.a) != 1:
            self.a.sort(key=lambda node: node._value, reverse=True)
            c = Node(value=(self.a[-1]._value + self.a[-2]._value))
            c._left = self.a.pop(-1)
            c._right = self.a.pop(-1)
            self.a.append(c)
        self.root = self.a[0]
        self.b = list(range(10))
        self.huffman_code = dict()

    def pre(self, tree, length):
        node = tree
        if not node:
            return
        elif node._name:
            self.huffman_code[node._name] = "".join(str(self.b[i]) for i in range(length))
            return
        self.b[length] = 0
        self.pre(node._left, length + 1)
        self.b[length] = 1
        self.pre(node._right, length + 1)

    def get_code(self):
        self.pre(self.root, 0)
        return self.huffman_code

#### Define Capacity Variables
capacity_best, capacity_average, capacity_worse = 0, 0, 0

## Define Dataset and DataLoader
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dir = './UCID1338'
test_dataset = MyData(test_dir, transforms_=dataset_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

## Image Preprocessing
dot = np.zeros((512, 512), dtype=np.float32)
cross = np.zeros((512, 512), dtype=np.float32)
hjy1 = np.zeros((512, 512), dtype=np.uint8)

for i in range(512):
    for j in range(512):
        if (i + j) % 2 == 0:
            dot[i][j] = 1
            hjy1[i][j] = 1
        else:
            cross[i][j] = 1
            hjy1[i][j] = 1

cross = np.expand_dims(cross, axis=2)
dot = np.expand_dims(dot, axis=2)
dot = dataset_transform(dot)
cross = dataset_transform(cross)

## Import and Load Model Properly
from NEW_CNN_5 import HJY_AC_CNNP

# Dynamically set the model file name based on train state
file_name = f'./Train_state{train_state}.pth'

# Check if the model file exists
if not os.path.exists(file_name):
    print(f"Error: Model file {file_name} not found!")
    exit(1)

# Initialize the model and move it to the MPS device
model = HJY_AC_CNNP().to(device)

# Load the saved weights properly
model.load_state_dict(torch.load(file_name, map_location=device))

# Set model to evaluation mode
model.eval()

## Prediction Loop
output_folder = f"predicted_images_method5_ts{train_state}"
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for ii, data in enumerate(test_dataloader):
        print(f"Processing Image {ii}")

        img, img_self, name = data
        img, img_self = img.to(device), img_self.to(device)

        img_self = torch.squeeze(img_self).cpu().numpy()

        img_dot = torch.mul(img, dot.to(device))
        img_cross = torch.mul(img, cross.to(device))

        ## Model Prediction
        predicted_image = model(img_dot)

        ## Post-Processing
        predicted_image = torch.squeeze(predicted_image).cpu().numpy()
        predicted_image = np.clip(np.around(predicted_image), 0, 255).astype(np.uint8)

        ## Save Predicted Image
        output_image_path = os.path.join(output_folder, name[0])
        cv2.imwrite(output_image_path, predicted_image)
        print(f"Saved: {os.path.abspath(output_image_path)}")

print("Processing completed for train state:", train_state)

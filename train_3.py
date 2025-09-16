import torchvision
from torch.utils.data import DataLoader
from Dataset_new import MyData
from NEW_CNN3 import HJY_AC_CNNP
import torch
from torch import nn
import numpy as np
import os

# Use MPS (Metal Performance Shaders) for Apple Silicon if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Dataset & Transformations
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

root_dir = './train_image'
test_dir = './ucid_gray'

# Custom DataLoader
train_dataset = MyData(root_dir, transforms_=dataset_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
test_dataset = MyData(test_dir, transforms_=dataset_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

# 2. Initialize Model
net = HJY_AC_CNNP().to(device)

# 3. Loss Function
loss_fn = nn.MSELoss().to(device)

# 4. Optimizer
learning_rate = 1e-3
weight_decay = 1e-3
optimizer = torch.optim.Adam(net.parameters(), weight_decay=weight_decay, lr=learning_rate)

# Training Parameters
total_train_step = 0
total_test_step = 0
epoch = 300

# Generate dot & cross pattern as input
dot = np.zeros((512, 512), dtype=np.float32)
cross = np.zeros((512, 512), dtype=np.float32)

for i in range(512):
    for j in range(512):
        if (i + j) % 2 == 0:
            dot[i][j] = 1
        else:
            cross[i][j] = 1

# Convert to Torch Tensors
cross = torch.tensor(cross).unsqueeze(0).to(device)  # Add channel dimension
dot = torch.tensor(dot).unsqueeze(0).to(device)      # Add channel dimension

# Training log file
txt = open('hjy_lognew.txt', 'a')

for i in range(epoch):
    print(f"_______________Epoch {i+1} Training Start________")
    txt.write(f"\n_______________Epoch {i+1} Training Start________")
    
    train_total_loss = 0
    net.train()

    for data in train_dataloader:
        img, _, _ = data
        img = img.to(device)
        img_dot = img * dot
        img_cross = img * cross

        output = net(img_dot)
        loss = loss_fn(output, img_cross)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        train_total_loss += loss.item()
        print(f"Step {total_train_step}, Loss: {loss.item()}")
        txt.write(f"\nStep {total_train_step}, Loss: {loss.item()}")

    print(f"Total Training Loss: {train_total_loss}")
    txt.write(f"\nTotal Training Loss: {train_total_loss}")

    # Validation
    net.eval()
    total_test_loss = 0

    with torch.no_grad():
        for data in test_dataloader:
            img, _, _ = data
            img = img.to(device)
            img_dot = img * dot
            img_cross = img * cross

            output = net(img_dot)
            loss = loss_fn(output, img_cross)
            total_test_loss += loss.item()
            total_test_step += 1

    print(f"Epoch {i+1}, Test Loss: {total_test_loss}")
    txt.write(f"\nEpoch {i+1}, Test Loss: {total_test_loss}")

    # Save Model Checkpoint Every 5 Epochs
    if i % 5 == 0:
        torch.save(net.state_dict(), f"Train_state{i}.pth")

txt.close()

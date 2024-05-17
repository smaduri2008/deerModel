import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from skimage import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class deer(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = torch.tensor([int(self.annotations.iloc[index, 1])], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        return (image, label)


transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = deer("data.csv", "./final_training_set", transform=transform)



class CNN(nn.Module):   #every image is 320 x 240
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 256, kernel_size=8) #in is 3 because we have 3 RGB layers, kernel and stride is the sizing of the convolutional filter.
    self.pool1 = nn.MaxPool2d(kernel_size = 8, stride = 8)

    self.conv2 = nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size=4)
    self.pool2 = nn.MaxPool2d(kernel_size = 4)

    self.conv3 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 2)
    self.pool3 = nn.MaxPool2d(kernel_size = 2)

    self.flatten = nn.Flatten()
    self.drop = nn.Dropout(0.5)

    self.fc1 = nn.Linear(1024*8, 2048)
    self.fc2 = nn.Linear(2048, 512)
    self.fc3 = nn.Linear(512, 128)

    self.fc4 = nn.Linear(128, 1)


  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    x = self.flatten(x)
    x = self.drop(F.relu(self.fc1(x)))
    x = self.drop(F.relu(self.fc2(x)))
    x = self.drop(F.relu(self.fc3(x)))
    x = self.fc4(x)
    x = F.sigmoid(x)
    return x



data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
epochs = 10

model = CNN()
model.to(device)
#storch.load("models/model.pth")
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
model.eval()

def train():
    for epoch in range(epochs):
        correct = 0
        for i, data in enumerate(data_loader):
            print(f"\rbatch #: {i}", end="\r")
            image, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            prediction = model(image)
            rounded = torch.round(prediction)
            correct += (rounded == label).sum().item()
            loss = criterion(prediction, label.float())
            loss.backward()
            optimizer.step()


        accuracy = correct / len(dataset) * 100
        print(f'epoch #{epoch+1}/{epochs} | accuracy: {accuracy}')


def test():
    for i in range(12):
        with torch.no_grad():
            extension = ".jpeg"
            if i == 10 or i == 11:
                extension = ".jpg"
            img_path = os.path.join("./dataset/final_test_set/", str(i + 1) + extension)
            image = io.imread(img_path)
            image = transforms.ToTensor()(image)
            image = image.view(1, 3, 240, 320).to(device)

            with torch.no_grad():
                predictions = model(image.to(device))
                predicted_class = predictions
                print(f'Predicted: {predicted_class}')
            title = "DEER DETECTED"
            if predicted_class[0][0] < 0.5:
                title = "DEER NOT DETECTED"
            plt.title(title)
            image_show = io.imread(img_path)
            plt.imshow(image_show)
            plt.show()






def saveModel():
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'model1.pth')
    torch.save(model.state_dict(), file_path)


train()



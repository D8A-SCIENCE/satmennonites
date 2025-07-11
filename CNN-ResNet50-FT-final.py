import random, os, glob
import sklearn.model_selection
import numpy as np

import geopandas as gpd
import pandas as pd


SAVE_MODEL_PTH='better-model-ResNet50-FT.pt'
NUM_EPOCHS=50
BATCH_SIZE=32
IMG_DIR = 'GridImgsDL'
np.random.seed(123)

###################################################################################################### 
################################## Weighted sampling to obtain test observations #####################
# read in the scruzGridGDF.shp file with grid geometries
gdf = gpd.read_file('data/scruzGridGDF.shp')
gdf = gdf.query("lcType=='CROPLAND' & gridClass in ['COMM1','OTHER']")

classCounts = dict(gdf['gridClass'].value_counts())
gdf['samplWeight'] = gdf['gridClass'].\
                        apply(lambda x: 1/(classCounts[x]/sum(classCounts.values())))

testGrids = gdf.sample(frac=0.10, weights='samplWeight')

print('test sample\n', dict(testGrids['gridClass'].value_counts()))

###################################################################################################### 
################################### train-validation split ###########################################

testLabels = testGrids.apply(
    lambda x: (f"{IMG_DIR}/{x['imgFP']}.npy",1) if x['gridClass']=='COMM1' else (f"{IMG_DIR}/{x['imgFP']}.npy",0),
    axis=1).to_list()

trainLabels = gdf.drop(index=testGrids.index, axis=0).apply(
    lambda x: (f"{IMG_DIR}/{x['imgFP']}.npy",1) if x['gridClass']=='COMM1' else (f"{IMG_DIR}/{x['imgFP']}.npy",0),
    axis=1).to_list()

random.shuffle(trainLabels)

trainSet, valSet = sklearn.model_selection.train_test_split(trainLabels, test_size=0.15)
testSet = testLabels

###########################################################################################################
################################ Create Dataloaders, transform data #######################################


import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image
import numpy as np
import time

def loadRaster(imgFP):
    with open(imgFP, 'rb') as f:
        imgArr = np.load(f, allow_pickle=False)
        # read only R,G,B bands
    return Image.fromarray(imgArr[:,:,[0,1,2]])


transform = transforms.Compose(
    [transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# transformations for Augmentation operations each for null and true class images
trueTransform = transforms.Compose([
    transforms.RandomRotation([179,180])])
nullTransform = transforms.Compose([
    transforms.RandomRotation([89,91], interpolation=transforms.InterpolationMode.BILINEAR)])

class CustomImageDataset(Dataset):
    def __init__(self, labels, transform, augment=False,factor=1):
        self.img_labels = labels
        self.transform = transform
        self.factor = factor
        self.augment = augment

    def __len__(self):
        return len(self.img_labels)*self.factor

    def __getitem__(self, idx):
        image = loadRaster(self.img_labels[idx][0])
        label = self.img_labels[idx][1]
        image = self.transform(image)
        # separate transformations for true and null classes
        if self.augment:
            if label==0:
                pass
            elif label==1:
                image = trueTransform(image)
        return image, label


# We're augmenting only the COMM class because the imbalance

augTrainDS = CustomImageDataset([(_,lab) for _,lab in trainSet if lab==1],
                                 transform,augment=True,factor=1)

trainDl = DataLoader(torch.utils.data.ConcatDataset([CustomImageDataset(trainSet,transform), augTrainDS]),
                     batch_size=BATCH_SIZE, shuffle=True)
# trainDl = DataLoader(CustomImageDataset(trainSet,transform),
#                      batch_size=BATCH_SIZE, shuffle=True)
valDl = DataLoader(CustomImageDataset(valSet, transform), 
                    batch_size=BATCH_SIZE, shuffle=True)
testDl = DataLoader(CustomImageDataset(testSet, transform), 
                     batch_size=BATCH_SIZE, shuffle=True)

# Print the class distribution for the custom dataset
print("Train dataset size by class:", 
        dict(Counter([label for _, label in CustomImageDataset(trainSet,transform)])))

#######################################################################################################
################## Instantiate ResNet50 model; create train setup #####################################

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(weights='IMAGENET1K_V1')

# Number of input features of last layer
# Number of output features: 'COMM' and 'OTHER'
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.50),
    nn.Linear(num_ftrs, 2))

loss_fn = nn.CrossEntropyLoss()

# All layer params are optimized in Fine tuning
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train(model, trainDl, valid_dl, loss_fn, optimizer, num_epochs):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in trainDl:     
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(trainDl.dataset)
        accuracy_hist_train[epoch] /= len(trainDl.dataset)

        # model.eval() is a built-in method inherited from nn.Module used to set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f} loss: {loss_hist_train[epoch]:.4f}')
    return model

model = model.to(device)
model = train(model, trainDl, valDl, loss_fn, optimizer, NUM_EPOCHS)
torch.save(model.state_dict(), SAVE_MODEL_PTH)

##################################################################################################################
####################### Model evaluation on test, train, validation data ########################################

import sklearn.metrics

## on test data
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for inputs, labels in testDl:
        outputs = model(inputs.to(device))
        
        pred = torch.argmax(outputs, dim=1)

        # we have to move back data to CPU
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

testAcc = sklearn.metrics.accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {testAcc:.4f}")

print('\n')

print('Per-class accuracy')
per_class = sklearn.metrics.classification_report(y_true, y_pred,target_names=['OTHER','COMM'])
print(per_class)
    
## forward pass on train data
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for inputs, labels in trainDl:
        outputs = model(inputs.to(device))
        
        pred = torch.argmax(outputs, dim=1)

        # we have to move back data to CPU
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

trainAcc = sklearn.metrics.accuracy_score(y_true, y_pred)
print(f"Train Accuracy: {trainAcc:.4f} \n")

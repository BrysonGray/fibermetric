#!/usr/bin/env python
# ruff: noqa: E501
"""
A U-Net model to predict the third component of the principal eigenvector given the 2x2 upper left submatrix of the diffusion
tensors from tensor valued image slices.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# first we'll make a dataset class
class DiffusionTensorDataset(Dataset):
    def __init__(self, img_file, mask_file=None):
        data = np.fromfile(img_file, dtype=np.float32)
        I_ = data.reshape(181,6,217,181)
        self.I = np.zeros((I_.shape[0], I_.shape[2], I_.shape[3],3,3))
        self.I[...,0,0] = I_[:,2]
        self.I[...,1,1] = I_[:,1]
        self.I[...,2,2] = I_[:,0]
        self.I[...,0,1] = I_[:,5]
        self.I[...,0,2] = I_[:,4]
        self.I[...,1,2] = I_[:,3]
        self.I[...,1,0] = I_[:,5]
        self.I[...,2,0] = I_[:,4]
        self.I[...,2,1] = I_[:,3]
        if mask_file is not None:
            mask = np.fromfile(mask_file, dtype=np.uint16)
            mask = mask.reshape(181,217,181)
            self.I[mask==0] = 0
        self.d,self.v = np.linalg.eigh(self.I) 

    def __len__(self):
        # return self.I.shape[0]*self.I.shape[1]*self.I.shape[2]
        return self.I.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(np.stack((self.I[idx,:,:,0,0], self.I[idx,:,:,1,1], self.I[idx,:,:,0,1])), dtype=torch.float),\
               torch.tensor(self.v[idx,...,-1,-1], dtype=torch.float).squeeze()


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # scale output between -1 and 1
            nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x).squeeze()


class UNet2D(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 1))
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)        
        x = self.up1(x5, x4)        
        x = self.up2(x, x3)        
        x = self.up3(x, x2)        
        x = self.up4(x, x1)

        return self.outc(x)
    
    
def train_loop(model, loss_fn, optimizer, train_loader, device):
    size = len(train_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return model

def test_loop(model, loss_fn, test_loader, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(test_loader)
    accuracy = []
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            print('Loss: ', loss_fn(pred, y).item())
            accuracy.append(torch.mean(torch.abs(pred-y)).item())
            # errors = torch.abs(pred-y).item()

    test_loss /= num_batches
    accuracy = np.mean(accuracy)
    print(f"Test Error: Avg loss: {test_loss:>8f} \n Accuracy: {accuracy:>8f} \n")
    
    return test_loss, accuracy

def main():
    """ Main function.
    """
    parser = argparse.ArgumentParser(description='Train a U-Net model for predicting out-of-plane fiber orientation from 2x2 tensor valued image slices.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to the training data.')
    parser.add_argument('-m', '--mask_dir', type=str, default=None, help='Path to the mask data.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('-n', '--name', type=str, required=True, help='Name of the model.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('-v', '--validation_steps', type=int, default=10, help='Validation steps.')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Path to the checkpoint file.')
    parser.add_argument('-p', '--patch_size', type=int, default=128, help='Patch size.')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('-u', '--unet_depth', type=int, default=4, help='U-Net depth.')
    parser.add_argument('-z', '--unet_filters', type=int, default=16, help='U-Net filters')

    args = parser.parse_args()


    # set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    files = os.listdir(args.input_dir)
    files.sort()
    if args.mask_dir is not None:
        masks = [mask for mask in os.listdir(args.mask_dir) if mask.endswith('.img')]
        masks.sort()
    else:
        masks = [None]*len(files)
    dataset = []
    print('Loading data...')
    for file, mask in zip(files[:-1], masks[:-1]):
        dataset.append(DiffusionTensorDataset(os.path.join(args.input_dir, file), os.path.join(args.mask_dir, mask)))
    dataset = torch.utils.data.ConcatDataset(dataset)
    print('Done')
    # split dataset into train and test
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    # validation_dataset, test_dataset = torch.utils.data.random_split(validation_dataset, [validation_size//2, validation_size-(validation_size//2)])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    model = UNet2D(n_channels=3).to(device)
    # This is a regression problem, so we'll use mean squared error as the loss function
    loss_fn = nn.MSELoss().to(device)
    # We'll use the Adam optimizer, which is a variant of stochastic gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # we'll use lr_scheduler to reduce the learning rate by a factor of 0.1 when the validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    losses = []
    accuracies = []
    print('Training...')
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        model = train_loop(model, loss_fn, optimizer, train_loader, device=device)
        loss, accuracy = test_loop(model, loss_fn, test_loader, device=device)
        losses.append(loss)
        accuracies.append(accuracy)
        scheduler.step(loss)
    print("Done")

    print("Saving...")
    # save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, args.name + '.pth'))
    # save losses and accuracies
    np.save(os.path.join(args.output_dir, 'losses.npy'), np.array(losses))
    np.save(os.path.join(args.output_dir, 'accuracies.npy'), np.array(accuracies))
    print("Done")

    return 0

if __name__ == '__main__':
    main()
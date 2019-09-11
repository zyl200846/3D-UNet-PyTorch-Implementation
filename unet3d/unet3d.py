# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import sys
import time
import torch
import torch.nn as nn
from unet3d.building_components import EncoderBlock, DecoderBlock
sys.path.append("..")


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        print("Final output shape: ", x.shape)
        return x


class Trainer(object):

    def __init__(self, data_dir, net, optimizer, criterion, no_epochs, batch_size=1):
        """
        Parameter initialization
        :param data_dir: folder that stores images for each modality
        :param net: the created model
        :param optimizer: the optimizer mode
        :param criterion: loss function
        :param no_epochs: number of epochs to train the model
        :param batch_size: batch size for generating data during training
        """
        self.data_dir = data_dir
        self.modalities = ["PET", "MASK"]
        self.net = net
        if torch.cuda.is_available():
            self.net.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.no_epochs = no_epochs
        self.batch_size = batch_size

    def train(self, data_paths_loader, dataset_loader, batch_data_loader):
        """
        Load corresponding data and start training
        :param data_paths_loader: get data paths ready for loading
        :param dataset_loader: get images and masks data
        :param batch_data_loader: generate batch data
        :return: None
        """
        self.net.train()
        pet_paths = data_paths_loader(self.data_dir, self.modalities[0])
        print(pet_paths)
        mask_paths = data_paths_loader(self.data_dir, self.modalities[1])
        pets, masks = dataset_loader(pet_paths, mask_paths)
        training_steps = len(pets) // self.batch_size

        for epoch in range(self.no_epochs):
            start_time = time.time()
            train_losses, train_iou = 0, 0
            for step in range(training_steps):
                self.net.zero_grad()

                x_batch, y_batch = batch_data_loader(pets, masks, iter_step=step, batch_size=1)
                x_batch = torch.from_numpy(x_batch).cuda()
                y_batch = torch.from_numpy(y_batch).cuda()

                logits = self.net(x_batch)
                print(logits.shape)
                loss = self.criterion(y_batch, logits)
                loss.backward()
                self.optimizer.step()
                # train_iou += mean_iou(y_batch, logits)
                train_losses += loss
            end_time = time.time()
            print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, train_losses / training_steps,
                                                                       start_time - end_time))

    def predict(self):
        pass

    def _save_checkpoint(self):
        pass


if __name__ == "__main__":
    inputs = torch.randn(1, 1, 96, 96, 96)
    print("The shape of inputs: ", inputs.shape)
    data_folder = "../processed"
    model = UnetModel(in_channels=1, out_channels=1)
    inputs = inputs.cuda()
    model.cuda()
    model(inputs)

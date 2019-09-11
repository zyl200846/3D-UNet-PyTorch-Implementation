# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 08:54
import torch
import torch.nn as nn
from unet3d_model.unet3d import UnetModel, Trainer
from unet3d_model.loss import DiceLoss
from data_gen import get_data_paths, data_gen, batch_data_gen


def train_main(data_folder, in_channels, out_channels, learning_rate, no_epochs):
    """
    Train module
    :param data_folder: data folder
    :param in_channels: the input channel of input images
    :param out_channels: the final output channel
    :param learning_rate: set learning rate for training
    :param no_epochs: number of epochs to train model
    :return:
    """
    model = UnetModel(in_channels=in_channels, out_channels=out_channels)
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = DiceLoss()
    trainer = Trainer(data_dir=data_folder, net=model, optimizer=optim, criterion=criterion, no_epochs=no_epochs)
    trainer.train(data_paths_loader=get_data_paths, dataset_loader=data_gen, batch_data_loader=batch_data_gen)


if __name__ == "__main__":
    data_dir = "./processed"
    train_main(data_folder=data_dir, in_channels=1, out_channels=1, learning_rate=0.001, no_epochs=5)

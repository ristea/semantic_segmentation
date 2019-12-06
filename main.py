import json

from torch import optim

from data.data_manager import DataManager
from trainer import Trainer
from utils.losses import SegmentationLosses
from visualizer import VisdomVisualizer
from networks.unet import UNet
from networks.unet_v2 import UNetV2
import torch


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0001)


def main():
    config = json.load(open('./config.json'))

    experiment_name = 'Fully Conv'
    vis_legend = ['Training Loss', 'Eval Loss']
    visualizer = VisdomVisualizer(experiment_name, vis_legend, config=config)

    model = UNetV2(n_channels=config['n_channels'], n_classes=config['n_classes'])
    model.apply(init_weights)

    weights = torch.ones(21)
    if config['use_cuda'] is True:
        weights = weights.cuda()
    weights[0] = 0.07
    weights[15] = 0.5

    criterion = SegmentationLosses(weight=weights).build_loss(mode='ce')
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.get_train_eval_dataloaders()

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, visualizer, experiment_name, config)
    trainer.train()


if __name__ == "__main__":
    main()

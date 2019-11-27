import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import utils.transforms as extended_transforms
from data.voc_dataset import VOC


class DataManager:
    def __init__(self, config):
        self.config = config
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_train_eval_dataloaders(self):
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*self.mean_std)
        ])
        target_transform = extended_transforms.MaskToTensor()

        # restore_transform = standard_transforms.Compose([
        #     extended_transforms.DeNormalize(*self.mean_std),
        #     standard_transforms.ToPILImage(),
        # ])
        #
        # visualize = standard_transforms.Compose([
        #     standard_transforms.Scale(400),
        #     standard_transforms.CenterCrop(400),
        #     standard_transforms.ToTensor()
        # ])

        train_set = VOC(self.config['data_path'], 'train', transform=input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], num_workers=8, shuffle=True)

        val_set = VOC(self.config['data_path'], 'val', transform=input_transform, target_transform=target_transform)
        val_loader = DataLoader(val_set, batch_size=self.config['batch_size'], num_workers=8, shuffle=False)

        return train_loader, val_loader

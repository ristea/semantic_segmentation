import torch
import json
import numpy as np
from PIL import Image

from data.data_manager import DataManager


def view_results():
    model = torch.load('../exp_results/model_epoch_70.pkl', map_location='cpu')
    config = json.load(open('../config.json'))

    model.eval()
    data_manager = DataManager(config)
    val_loader, _ = data_manager.get_train_eval_dataloaders()

    for idx, (inputs, labels) in enumerate(val_loader, 0):
        if config['use_cuda']:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()
        else:
            inputs = inputs.float()
            labels = labels.float()

        predictions = model(inputs)

        predictions = predictions.data.cpu().numpy()
        labels = labels.cpu().numpy()

        predictions = np.argmax(predictions, axis=1)

        img = Image.fromarray(labels[0])
        img.show()
        print(predictions)


if __name__ == "__main__":
    view_results()










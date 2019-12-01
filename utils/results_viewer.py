import torch
import json
import numpy as np
from PIL import Image

from data.data_manager import DataManager


def create_mask(labels, palette):
    h, w = labels.shape
    img = np.zeros((h, w, 3))
    for row in range(h):
        for col in range(w):
            l = int(labels[row][col])
            r = l * 3
            g = l * 3 + 1
            b = l * 3 + 2
            try:
                img[row][col] = (palette[r], palette[g], palette[b])
            except:
                img[row][col] = (255, 255, 255)
    return img


def get_k_predictions(predictions, k):
    pred1 = np.argmax(predictions, axis=0)
    predictions[pred1] = float('inf')
    pred2 = np.argmax(predictions, axis=0)
    predictions[pred2] = float('inf')
    pred3 = np.argmax(predictions, axis=0)
    predictions[pred3] = float('inf')
    return pred1, pred2, pred3


def view_results():
    model = torch.load('../temp/exp_results/model_epoch_100.pkl', map_location='cpu')
    config = json.load(open('../config.json'))

    model.eval()
    data_manager = DataManager(config)
    val_loader, _ = data_manager.get_train_eval_dataloaders()

    for idx, (inputs, labels) in enumerate(val_loader, 0):
        print(idx)
        if config['use_cuda']:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()
            model = model.cuda()
        else:
            inputs = inputs.float()
            labels = labels.float()

        predictions = model(inputs)

        predictions = predictions.data.cpu().numpy()
        print('predictions', np.unique(predictions[0]))
        labels = labels.cpu().numpy()
        print('labels', np.unique(labels[0]))
        mask = create_mask(labels[0], palette=config['palette'])
        img = Image.fromarray(mask.astype(np.uint8))
        img.save('gt' + str(idx) + '.jpg')

        predictions = np.argmax(predictions[0], axis=0)
        mask = create_mask(predictions, palette=config['palette'])
        img = Image.fromarray(mask.astype(np.uint8))
        img.save('pred_' + '_' + str(idx) + '.jpg')



if __name__ == "__main__":
    view_results()










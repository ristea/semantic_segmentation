from visdom import Visdom
from utils.results_viewer import create_mask
import numpy as np
import torch


class VisdomVisualizer:
    def __init__(self, plot_title, vis_legend, _xlabel='Epoch', _ylabel='Loss', config=None):
        self.config = config
        # For getting rid of spikes
        self.last_loss1_value = 0
        self.last_loss2_value = 0

        self.viz = Visdom()
        self.plot_title = plot_title
        self.vis_legend = vis_legend
        self.window = self.viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel=_xlabel,
                ylabel=_ylabel,
                title=self.plot_title,
                legend=self.vis_legend
            )
        )

    def update_statistics(self, iteration, loss1, loss2, update_type='append', epoch_size=1):
        if loss1 is None:
            loss1 = self.last_loss1_value
        if loss2 is None:
            loss2 = self.last_loss2_value

        self.viz.line(
            X=torch.ones((1, 2)).cpu() * iteration,
            Y=torch.Tensor([loss1, loss2]).unsqueeze(0).cpu() / epoch_size,
            win=self.window,
            update=update_type
        )

        if loss1 != 0:
            self.last_loss1_value = loss1
        if loss2 != 0:
            self.last_loss2_value = loss2

    def update_images(self, prediction, label, eval=False):
        pred = prediction.detach().cpu().numpy()[0]
        pred = np.argmax(pred, axis=0)

        label = label.detach().cpu().numpy()[0]

        label = create_mask(label, palette=self.config['palette'])
        pred = create_mask(pred, palette=self.config['palette'])

        if eval is True:
            self.viz.image(np.moveaxis(label, 2, 0), win='Eval Label')
            self.viz.image(np.moveaxis(pred, 2, 0), win='Eval Prediction')

        self.viz.image(np.moveaxis(label, 2, 0), win='Label')
        self.viz.image(np.moveaxis(pred, 2, 0), win='Prediction')

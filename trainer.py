import torch
import numpy as np
import os
from tqdm import tqdm
from utils.metrics import Evaluator


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer, visualizer, experiment_name,
                 config):
        self.config = config
        self.network = network
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.visualizer = visualizer
        self.experiment_name = experiment_name
        self.evaluator = Evaluator(config['n_classes'])

    def train_epoch(self, epoch):
        running_loss = []
        for idx, (inputs, labels) in enumerate(self.train_dataloader, 0):
            self.network.train()
            if self.config['use_cuda']:
                inputs = inputs.cuda().float()
                labels = labels.cuda().float()
            else:
                inputs = inputs.float()
                labels = labels.float()

            self.optimizer.zero_grad()
            predictions = self.network(inputs)

            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                self.visualizer.update_statistics(idx + len(self.train_dataloader) * epoch,
                                                  loss1=running_loss, loss2=None)
                print(f'Training loss on iteration {idx} = {running_loss}')
                running_loss = []

    def eval_net(self, epoch):
        running_eval_loss = 0.0
        self.network.eval()
        for i, (inputs_, labels_) in enumerate(self.eval_dataloader, 0):
            if self.config['use_cuda']:
                inputs_ = inputs_.cuda().float()
                labels_ = labels_.cuda().float()
            else:
                inputs_ = inputs_.float()
                labels_ = labels_.float()

            predictions_ = self.network(inputs_)
            eval_loss = self.criterion(predictions_, labels_)
            running_eval_loss += eval_loss.item()

        running_eval_loss = running_eval_loss / len(self.eval_dataloader)
        self.visualizer.update_statistics(iteration=len(self.train_dataloader) * (epoch + 1),
                                          loss1=None, loss2=running_eval_loss)
        print(f'### Evaluation loss on epoch {epoch} = {running_eval_loss}')

    def train(self):
        try:
            os.mkdir(os.path.join(self.config['exp_path'], self.experiment_name))
        except FileExistsError:
            print("Director already exists! It will be overwritten!")

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

    def save_net_state(self, epoch):
        path_to_save = os.path.join(self.config['exp_path'], self.experiment_name, 'model_epoch_' + str(epoch) + '.pkl')
        torch.save(self.network, path_to_save)

    def validation(self, epoch):
        self.network.eval()
        self.evaluator.reset()

        tbar = tqdm(self.eval_dataloader, desc='\r')
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.config['use_cuda']:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.network(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d]' % epoch)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

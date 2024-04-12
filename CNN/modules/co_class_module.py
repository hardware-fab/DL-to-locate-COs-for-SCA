"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

import neptune

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import torchmetrics
from torchmetrics import F1Score as F1

import CNN.models as models


class CoClassifier(pl.LightningModule):
    def __init__(self, module_config, gpu=0):
        super(CoClassifier, self).__init__()

        # Build Model
        # -----------
        model_name = module_config['model']['name']
        model_config = module_config['model']['config']
        model_class = getattr(models, model_name)
        self.model = model_class(**model_config)
        # -----------

        # Define Loss
        # -----------
        loss_name = module_config['loss']['name']
        if 'config' in module_config['loss']:
            loss_config = module_config['loss']['config']
            if 'weight' in loss_config and 'Entropy' in loss_name:
                weights = loss_config['weight']
                class_weights = torch.FloatTensor(
                    weights).to(torch.device(f'cuda:{gpu}'))
                loss_config['weight'] = class_weights
        else:
            loss_config = {}
        loss_class = getattr(nn, loss_name)
        self.loss = loss_class(**loss_config)
        # -----------

        # Get Optimization Config
        # -----------------------
        self.optimizer_config = module_config['optimizer']
        if 'scheduler' in module_config:
            self.scheduler_config = module_config['scheduler']
        else:
            self.scheduler_config = None
        # -----------------------

        # Get metrics
        # -----------
        self.metrics = {}

        metric_configs = module_config['metrics']
        for metric_config in metric_configs:
            metric_name = metric_config['name']
            metric_conf = metric_config['config']
            if 'F1' in metric_name:
                metric_class = F1
            else:
                metric_class = getattr(torchmetrics, metric_name)
            metric = metric_class(**metric_conf)
            self.metrics[metric_name] = metric.to(torch.device(f'cuda:{gpu}'))
        # -----------

        self.original_labels_train = []
        self.original_labels_val = []
        self.original_labels_test = []

    def forward(self, x):
        return self.model(x)

    def step_batch(self, batch):
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return {'loss': loss, 'target': y.detach(), 'prediction': nn.functional.softmax(y_hat, dim=1).detach()}

    def training_step(self, batch, batch_idx):
        return self.step_batch(batch)

    def validation_step(self, batch, batch_idx):
        return self.step_batch(batch)

    def test_step(self, batch, batch_idx):
        return self.step_batch(batch)

    def validation_epoch_end(self, outputs):
        self.log_confusion_matrix(outputs)

    def test_epoch_end(self, outputs):
        self.log_confusion_matrix(outputs, 'test')

    def training_step_end(self, outputs):
        self.log("train/loss", outputs['loss'],
                 on_step=True, on_epoch=True, sync_dist=True)
        for metric_name in self.metrics:
            val = self.metrics[metric_name](
                outputs['prediction'], outputs['target'])
            self.log('train/{}'.format(metric_name), val,
                     on_step=True, on_epoch=True, sync_dist=True)

    def validation_step_end(self, outputs):
        self.log("valid/loss", outputs['loss'],
                 on_step=True, on_epoch=True, sync_dist=True)

        for metric_name in self.metrics:
            val = self.metrics[metric_name](
                outputs['prediction'], outputs['target'])
            self.log('valid/{}'.format(metric_name), val,
                     on_step=True, on_epoch=True, sync_dist=True)

    def log_confusion_matrix(self, outputs, which_subset='valid'):
        target = torch.cat([output['target'] for output in outputs])
        softmax_preds = torch.cat([output['prediction'] for output in outputs])
        preds = torch.argmax(softmax_preds, dim=1)

        target = target.cpu().data.numpy()
        preds = preds.cpu().data.numpy()

        fig, axis = plt.subplots(figsize=(16, 12))
        ConfusionMatrixDisplay.from_predictions(target, preds, ax=axis)
        #plot_confusion_matrix(target, preds, ax=axis)
        self.logger.experiment['{}/confusion_matrix'.format(which_subset)].log(
            neptune.types.File.as_image(fig))

    def test_step_end(self, outputs):
        self.log("test/loss", outputs['loss'],
                 on_step=True, on_epoch=True, sync_dist=True)

        for metric_name in self.metrics:
            val = self.metrics[metric_name](
                outputs['prediction'], outputs['target'])
            self.log('test/{}'.format(metric_name), val,
                     on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_name = self.optimizer_config['name']
        optimizer_config = self.optimizer_config['config']
        optimizer_class = getattr(torch.optim, optimizer_name)

        optimizer = optimizer_class(
            self.parameters(), **optimizer_config)
        if self.scheduler_config is None:
            return optimizer
        else:
            scheduler_name = self.scheduler_config['name']
            scheduler_config = self.scheduler_config['config']
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            scheduler = scheduler_class(
                optimizer, **scheduler_config)
            sched = {'scheduler': scheduler, 'monitor': 'valid/loss'}
            return {'optimizer': optimizer, 'lr_scheduler': sched}

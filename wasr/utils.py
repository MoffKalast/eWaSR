import os
import torch
import pytorch_lightning as pl

def load_weights(path):
    state_dict = torch.load(path, map_location='cpu')
    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']

    return state_dict

class ModelExporter(pl.Callback):
    """Exports model weights at the end of the training and optionally every n epochs."""
    def __init__(self, every_n_epochs=0, monitor=None, mode='min'):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _export(self, trainer, pl_module, filename):
        # Trainer.log_dir broadcasts across ranks, so every rank has to reach it. Resolving it
        # inside the rank-zero guard below would leave rank zero waiting on a collective the
        # other ranks never join, which deadlocks DDP.
        log_dir = trainer.log_dir
        if not trainer.is_global_zero or log_dir is None:
            return
        torch.save(pl_module.model.state_dict(), os.path.join(log_dir, filename))

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs > 0 and epoch % self.every_n_epochs == 0:
            self._export(trainer, pl_module, 'weights_epoch%03d.pth' % epoch)

    def on_validation_end(self, trainer, pl_module):
        if self.monitor is None or trainer.sanity_checking or self.monitor not in trainer.callback_metrics:
            return

        value = trainer.callback_metrics[self.monitor].item()
        if self.best is None or (value < self.best if self.mode == 'min' else value > self.best):
            self.best = value
            self._export(trainer, pl_module, 'best.pth')

    def on_fit_end(self, trainer, pl_module):
        self._export(trainer, pl_module, 'weights.pth')

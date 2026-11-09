from PIL import Image

import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from .loss import focal_loss, water_obstacle_separation_loss
from .metrics import PixelAccuracy, ClassIoU, MeanIoU
from datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
LR_DECAY_POW = 0.9
FOCAL_LOSS_SCALE = 'labels'
SL_LAMBDA = 0.01
LOG_IMAGES_EVERY = 5
VAL_IMAGES = 8
PALETTE = torch.tensor([[247, 195, 37], [41, 167, 224], [90, 75, 164], [0, 0, 0]], dtype=torch.uint8)

def colorize(classes):
    return PALETTE.to(classes.device)[classes.clamp(max=3)].permute(2, 0, 1)

def denormalize(image):
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return ((image * std + mean) * 255).clamp(0, 255).byte()

class LitModel(pl.LightningModule):
    """ Pytorch Lightning wrapper for a model, ready for distributed training. """

    @staticmethod
    def add_argparse_args(parser):
        """Adds model specific parameters to parser."""

        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                            help="Number of training epochs.")
        parser.add_argument("--lr_decay_pow", type=float, default=LR_DECAY_POW,
                            help="Decay parameter to compute the learning rate decay.")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--focal_loss_scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'],
                            help="Which scale to use for focal loss computation (logits or labels).")
        parser.add_argument("--no_separation_loss", action='store_true', help="Disable separation loss.")
        parser.add_argument("--separation_loss_lambda", default=SL_LAMBDA, type=float,
                            help="The separation loss lambda (weight).")
        parser.add_argument("--mixer", type=str, default="CCCCSS", help="Token mixers in feature mixer.")
        #parser.add_argument("-l", "--transformer_blocks", type=int, default=6, help="Number of transformer blocks in TopFormer")
        #parser.add_argument("--skip", type=str, default=None, choices=[None, 'arm', 'sarm'], help="Use 2 transformer blocks on SKIP connection")
        #parser.add_argument("--short", type=str, default=None, choices=[None, 'arm', 'sarm'], help="Last two of L blocks type")
        parser.add_argument("--project", action='store_true', help="Project encoder features to less channels.")
        #parser.add_argument("--ablation", type=str, default=None, choices=["noarm1","noarm2", "noaspp", "noaspp1", "noffm", "noffm1"])
        parser.add_argument("--enricher", type=str, default="SS", help="Token mixers in long-skip feature enricher.")
        parser.add_argument("--pyramid", action='store_true', help="Run the segmentation head at half resolution, fed by the backbone stem and a Laplacian pyramid level.")
        parser.add_argument("--log_images_every", type=int, default=LOG_IMAGES_EVERY, help="Log fixed validation predictions to TensorBoard every n epochs (0 disables, also disables logging the first training batches).")
    
        return parser

    def __init__(self, model, num_classes, args, val_widths=()):
        super().__init__()

        self.model = model
        self.num_classes = num_classes

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.lr_decay_pow = args.lr_decay_pow
        self.focal_loss_scale = args.focal_loss_scale
        self.separation_loss = not args.no_separation_loss
        self.separation_loss_lambda = args.separation_loss_lambda
        self.log_images_every = args.log_images_every
        self.weights_before = None

        # Metrics
        self.val_accuracy = PixelAccuracy(num_classes)
        self.val_iou_0 = ClassIoU(0, num_classes)
        self.val_iou_1 = ClassIoU(1, num_classes)
        self.val_iou_2 = ClassIoU(2, num_classes)
        self.val_miou_by_width = torch.nn.ModuleDict({str(w): MeanIoU(num_classes) for w in val_widths})
        self.train_miou = MeanIoU(num_classes)

    def forward(self, x):
        output = self.model(x)
        return output['out']

    def training_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        fl = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        if self.separation_loss:
            separation_loss = water_obstacle_separation_loss(out['aux'], labels['segmentation'])
        else:
            separation_loss = torch.tensor(0.0)

        separation_loss = self.separation_loss_lambda * separation_loss
        loss = fl + separation_loss

        # log losses
        self.log('train/loss', loss.item())
        self.log('train/focal_loss', fl.item())
        self.log('train/separation_loss', separation_loss.item())

        with torch.no_grad():
            preds, labels_hard = self._hard_predictions(out['out'], labels['segmentation'])
            self.train_miou(preds, labels_hard)
            self.log('train/miou', self.train_miou, on_step=False, on_epoch=True)

        if self.log_images_every > 0 and self.current_epoch == 0 and batch_idx < 4 and self.trainer.is_global_zero:
            panels = [torch.cat([denormalize(image), colorize(target)], dim=2) for image, target in zip(features['image'], labels_hard)]
            self.logger.experiment.add_image(f'train_batches/{batch_idx}', torch.cat(panels, dim=1), self.global_step)

        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.trainer.is_global_zero and self.global_step % self.trainer.log_every_n_steps == 0:
            self.weights_before = [(group['name'], [(p, p.detach().clone()) for p in group['params']]) for group in optimizer.param_groups]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.weights_before is None:
            return

        for name, params in self.weights_before:
            update = sum((p.detach() - before).pow(2).sum() for p, before in params).sqrt()
            weight = sum(before.pow(2).sum() for _, before in params).sqrt()
            self.logger.experiment.add_scalar(f'debug/update_ratio/{name}', (update / weight).item(), self.global_step)

        self.weights_before = None

    def _hard_predictions(self, logits, segmentation):
        labels_size = (segmentation.size(2), segmentation.size(3))
        logits = TF.resize(logits, labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)

        # Create hard labels from soft
        labels_hard = segmentation.argmax(1)
        ignore_mask = segmentation.sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        return preds, labels_hard

    # def training_epoch_end(self, outputs):
    #     # Bugfix
    #     pass

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        loss = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        # Log loss
        self.log('val/loss', loss, batch_size=features['image'].size(0), sync_dist=True)

        # Metrics
        preds, labels_hard = self._hard_predictions(out['out'], labels['segmentation'])

        self.val_accuracy(preds, labels_hard)
        self.val_iou_0(preds, labels_hard)
        self.val_iou_1(preds, labels_hard)
        self.val_iou_2(preds, labels_hard)

        self.log('val/accuracy', self.val_accuracy)
        self.log('val/iou/obstacle', self.val_iou_0)
        self.log('val/iou/water', self.val_iou_1)
        self.log('val/iou/sky', self.val_iou_2)

        width = str(features['image'].size(3))
        self.val_miou_by_width[width](preds, labels_hard)
        self.log(f'val/miou_by_width/{width}', self.val_miou_by_width[width])

        if self._log_images_now() and batch_idx % max(1, self.trainer.num_val_batches[0] // VAL_IMAGES) == 0:
            panel = torch.cat([features['image_original'][0], colorize(labels_hard[0]), colorize(preds[0])], dim=2)
            self.logger.experiment.add_image(f'val_predictions/{batch_idx:03d}', panel, self.current_epoch + 1)

        return {'loss': loss, 'preds': preds}

    def on_validation_epoch_end(self):
        miou = (self.val_iou_0.compute() + self.val_iou_1.compute() + self.val_iou_2.compute()) / 3
        self.log('val/miou', miou)

    def _log_images_now(self):
        return self.log_images_every > 0 and not self.trainer.sanity_checking and self.trainer.is_global_zero and (self.current_epoch + 1) % self.log_images_every == 0

    def configure_optimizers(self):
        # Separate parameters for different LRs
        encoder_parameters = []
        decoder_w_parameters = []
        decoder_b_parameters = []
        for name, parameter in self.model.named_parameters():
            if name.startswith('backbone'):
                encoder_parameters.append(parameter)
            elif 'weight' in name:
                decoder_w_parameters.append(parameter)
            else:
                decoder_b_parameters.append(parameter)

        optimizer = torch.optim.RMSprop([
            {'params': encoder_parameters, 'lr': self.learning_rate, 'name': 'encoder'},
            {'params': decoder_w_parameters, 'lr': self.learning_rate * 10, 'name': 'decoder_weights'},
            {'params': decoder_b_parameters, 'lr': self.learning_rate * 20, 'name': 'decoder_biases'},
        ], momentum=self.momentum, alpha=0.9, weight_decay=self.weight_decay)

        # Decaying LR function
        lr_fn = lambda epoch: (1 - epoch/self.epochs) ** self.lr_decay_pow

        # Decaying learning rate (updated each epoch)
        scheduler = LambdaLR(optimizer, lr_fn)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Export the model weights
        checkpoint['model'] = self.model.state_dict()

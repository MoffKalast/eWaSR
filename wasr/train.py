import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from .loss import focal_loss, water_obstacle_separation_loss
from .metrics import PixelAccuracy, ClassIoU, MeanIoU
from datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

NUM_EPOCHS = 100
LEARNING_RATE = 6e-5
DECODER_LR_MULT = 10
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1500
LR_DECAY_POW = 1.0
DROP_PATH = 0.1
CH_SIM = 256
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
                            help="Encoder learning rate (AdamW) at the end of warmup.")
        parser.add_argument("--decoder_lr_mult", type=float, default=DECODER_LR_MULT,
                            help="Decoder learning rate as a multiple of the encoder one.")
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                            help="Number of training epochs.")
        parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                            help="Linear learning rate warmup length in optimizer steps.")
        parser.add_argument("--lr_decay_pow", type=float, default=LR_DECAY_POW,
                            help="Power of the per-step polynomial learning rate decay after warmup.")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="AdamW weight decay, not applied to norms, biases and layer scales.")
        parser.add_argument("--drop_path", type=float, default=DROP_PATH,
                            help="Maximum stochastic depth rate in the encoder.")
        parser.add_argument("--ch_sim", type=int, default=CH_SIM,
                            help="Decoder width. The main decoder cost knob.")
        parser.add_argument("--focal_loss_scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'],
                            help="Which scale to use for focal loss computation (logits or labels).")
        parser.add_argument("--no_separation_loss", action='store_true', help="Disable separation loss.")
        parser.add_argument("--separation_loss_lambda", default=SL_LAMBDA, type=float,
                            help="The separation loss lambda (weight).")
        parser.add_argument("--mixer", type=str, default="CCCCSS", help="Token mixers in feature mixer.")
        parser.add_argument("--enricher", type=str, default="SS", help="Token mixers in long-skip feature enricher.")
        parser.add_argument("--log_images_every", type=int, default=LOG_IMAGES_EVERY, help="Log fixed validation predictions to TensorBoard every n epochs (0 disables, also disables logging the first training batches).")
    
        return parser

    def __init__(self, model, num_classes, args, val_widths=()):
        super().__init__()

        self.model = model
        self.num_classes = num_classes

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.decoder_lr_mult = args.decoder_lr_mult
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
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

    def forward(self, image):
        return self.model(image)['out']

    def training_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features['image'])

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
        logits = TF.resize(logits, labels_size, interpolation=InterpolationMode.BILINEAR)
        preds = logits.argmax(1)

        # Create hard labels from soft
        labels_hard = segmentation.argmax(1)
        ignore_mask = segmentation.sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        return preds, labels_hard

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features['image'])

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
        groups = []
        for part, lr in (('encoder', self.learning_rate), ('decoder', self.learning_rate * self.decoder_lr_mult)):
            params = list(getattr(self.model, part).parameters())
            groups.append({'params': [p for p in params if p.ndim > 1], 'lr': lr, 'weight_decay': self.weight_decay, 'name': part})
            groups.append({'params': [p for p in params if p.ndim <= 1], 'lr': lr, 'weight_decay': 0.0, 'name': f'{part}_no_decay'})

        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.warmup_steps, total_steps // 2)

        def lr_fn(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            return max(0.0, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps)) ** self.lr_decay_pow

        scheduler = LambdaLR(optimizer, lr_fn)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_save_checkpoint(self, checkpoint):
        # Export the model weights
        checkpoint['model'] = self.model.state_dict()

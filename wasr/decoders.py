import torch
from torch import nn
import torch.nn.functional as F
from .layers import SIM, SegHead
from .metaformer import MetaFormerBlock, get_token_mixer

class EWaSRDecoder(nn.Module):

	def __init__(self, channels, num_classes=3, ch_sim=256, mixer='CCCCSS', enricher='SS'):
		super().__init__()
		self.channels = channels
		self.mixer = nn.Sequential(*[MetaFormerBlock(sum(channels), token_mixer=get_token_mixer(letter)) for letter in mixer])
		self.enricher = nn.Sequential(*[MetaFormerBlock(channels[1], token_mixer=get_token_mixer(letter)) for letter in enricher])
		self.sims = nn.ModuleList([SIM(c, ch_sim) for c in channels])
		self.seg_head = SegHead(ch_sim, num_classes)

	def forward(self, features):
		h, w = features[-1].shape[-2:]
		size = ((h - 1) // 2 + 1, (w - 1) // 2 + 1)
		tokens = torch.cat([F.adaptive_avg_pool2d(f, size) for f in features], dim=1)
		tokens = self.mixer(tokens).split(self.channels, dim=1)

		features = list(features)
		features[1] = self.enricher(features[1])

		size = features[0].shape[-2:]
		x = sum(F.interpolate(sim(f, t), size=size, mode='bilinear', align_corners=False) for sim, f, t in zip(self.sims, features, tokens))

		return self.seg_head(x)

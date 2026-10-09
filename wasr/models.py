from torch import nn
from .mit import MixTransformer
from .decoders import EWaSRDecoder

class EWaSR(nn.Module):

	def __init__(self, variant='b0', num_classes=3, pretrain='ade', drop_path=0.0, ch_sim=256, mixer='CCCCSS', enricher='SS'):
		super().__init__()
		self.encoder = MixTransformer(variant, drop_path)

		if pretrain is not None:
			self.encoder.load_pretrained(pretrain)

		self.decoder = EWaSRDecoder(self.encoder.channels, num_classes, ch_sim, mixer, enricher)

	def forward(self, image):
		features = self.encoder(image)
		return {'out': self.decoder(features), 'aux': features[2]}

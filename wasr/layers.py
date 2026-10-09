from torch import nn
import torch.nn.functional as F

class AttentionRefinementModule(nn.Module):

	def __init__(self, in_channels):
		super().__init__()
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		weights = self.sigmoid(self.bn1(self.conv1(self.global_pool(x))))
		return weights * x


class SIM(nn.Module):

	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.conv_lc = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_lc = nn.BatchNorm2d(ch_out)

		self.conv_gc = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_gc = nn.BatchNorm2d(ch_out)
		self.sigmoid_gc = nn.Sigmoid()

		self.conv_gc1 = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_gc1 = nn.BatchNorm2d(ch_out)

	def forward(self, lx, gx):
		size = lx.shape[-2:]
		lx = self.bn_lc(self.conv_lc(lx))
		multi = F.interpolate(self.sigmoid_gc(self.bn_gc(self.conv_gc(gx))), size=size, mode='nearest')
		gx = F.interpolate(self.bn_gc1(self.conv_gc1(gx)), size=size, mode='nearest')

		return lx * multi + gx


class SegHead(nn.Module):

	def __init__(self, ch, num_classes):
		super().__init__()
		self.conv = nn.Conv2d(ch, ch, 1)
		self.bn = nn.BatchNorm2d(ch)
		self.relu = nn.ReLU6()
		self.conv1 = nn.Conv2d(ch, num_classes, 1)

	def forward(self, x):
		return self.conv1(self.relu(self.bn(self.conv(x))))

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

VARIANTS = {
	'b0': {'channels': [32, 64, 160, 256], 'depths': [2, 2, 2, 2]},
	'b1': {'channels': [64, 128, 320, 512], 'depths': [2, 2, 2, 2]}
}

HEADS = [1, 2, 5, 8]
SR_RATIOS = [8, 4, 2, 1]
MLP_RATIO = 4

PRETRAINED_REPOS = {
	'imagenet': 'mit-{variant}',
	'ade': 'segformer-{variant}-finetuned-ade-512-512',
	'cityscapes': 'segformer-{variant}-finetuned-cityscapes-1024-1024'
}

def drop_path(x, p, training):
	if p == 0.0 or not training:
		return x

	keep = torch.empty((x.size(0),) + (1,) * (x.dim() - 1), device=x.device, dtype=x.dtype).bernoulli_(1 - p)
	return x * keep / (1 - p)


class DropPath(nn.Module):

	def __init__(self, p=0.0):
		super().__init__()
		self.p = p

	def forward(self, x):
		return drop_path(x, self.p, self.training)


class OverlapPatchEmbed(nn.Module):

	def __init__(self, ch_in, ch_out, kernel_size, stride):
		super().__init__()
		self.proj = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=kernel_size // 2)
		self.layer_norm = nn.LayerNorm(ch_out)

	def forward(self, x):
		x = self.proj(x)
		h, w = x.shape[-2:]
		x = self.layer_norm(x.flatten(2).transpose(1, 2))
		return x, h, w


class EfficientSelfAttention(nn.Module):

	def __init__(self, dim, heads, sr_ratio):
		super().__init__()
		self.heads = heads
		self.query = nn.Linear(dim, dim)
		self.key = nn.Linear(dim, dim)
		self.value = nn.Linear(dim, dim)
		self.sr_ratio = sr_ratio

		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
			self.layer_norm = nn.LayerNorm(dim)

	def split_heads(self, x):
		b, n, c = x.shape
		return x.view(b, n, self.heads, c // self.heads).transpose(1, 2)

	def forward(self, x, h, w):
		b, n, c = x.shape
		q = self.split_heads(self.query(x))

		if self.sr_ratio > 1:
			x = self.sr(x.transpose(1, 2).reshape(b, c, h, w))
			x = self.layer_norm(x.flatten(2).transpose(1, 2))

		k = self.split_heads(self.key(x))
		v = self.split_heads(self.value(x))

		x = F.scaled_dot_product_attention(q, k, v)
		return x.transpose(1, 2).reshape(b, n, c)


class AttentionOutput(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.dense = nn.Linear(dim, dim)

	def forward(self, x):
		return self.dense(x)


class Attention(nn.Module):

	def __init__(self, dim, heads, sr_ratio):
		super().__init__()
		self.self = EfficientSelfAttention(dim, heads, sr_ratio)
		self.output = AttentionOutput(dim)

	def forward(self, x, h, w):
		return self.output(self.self(x, h, w))


class DepthwiseConv(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

	def forward(self, x, h, w):
		b, n, c = x.shape
		x = self.dwconv(x.transpose(1, 2).reshape(b, c, h, w))
		return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):

	def __init__(self, dim, hidden):
		super().__init__()
		self.dense1 = nn.Linear(dim, hidden)
		self.dwconv = DepthwiseConv(hidden)
		self.dense2 = nn.Linear(hidden, dim)

	def forward(self, x, h, w):
		x = self.dense1(x)
		x = F.gelu(self.dwconv(x, h, w))
		return self.dense2(x)


class Block(nn.Module):

	def __init__(self, dim, heads, sr_ratio, drop_path):
		super().__init__()
		self.layer_norm_1 = nn.LayerNorm(dim)
		self.attention = Attention(dim, heads, sr_ratio)
		self.layer_norm_2 = nn.LayerNorm(dim)
		self.mlp = MixFFN(dim, dim * MLP_RATIO)
		self.drop_path = DropPath(drop_path)

	def forward(self, x, h, w):
		x = x + self.drop_path(self.attention(self.layer_norm_1(x), h, w))
		x = x + self.drop_path(self.mlp(self.layer_norm_2(x), h, w))
		return x


class MixTransformer(nn.Module):

	def __init__(self, variant='b0', drop_path=0.0):
		super().__init__()
		config = VARIANTS[variant]
		self.variant = variant
		self.channels = config['channels']

		ch_in = [3] + self.channels[:-1]
		self.patch_embeddings = nn.ModuleList([OverlapPatchEmbed(ch_in[i], c, 7 if i == 0 else 3, 4 if i == 0 else 2) for i, c in enumerate(self.channels)])

		rates = torch.linspace(0, drop_path, sum(config['depths'])).tolist()
		self.block = nn.ModuleList()
		for i, (c, depth) in enumerate(zip(self.channels, config['depths'])):
			stage_rates = rates[sum(config['depths'][:i]):sum(config['depths'][:i + 1])]
			self.block.append(nn.ModuleList([Block(c, HEADS[i], SR_RATIOS[i], r) for r in stage_rates]))

		self.layer_norm = nn.ModuleList([nn.LayerNorm(c) for c in self.channels])

	def load_pretrained(self, source):
		repo = PRETRAINED_REPOS[source].format(variant=self.variant)
		url = f'https://huggingface.co/nvidia/{repo}/resolve/main/pytorch_model.bin'
		state_dict = load_state_dict_from_url(url, file_name=f'{repo}.bin', map_location='cpu', progress=True)

		prefix = 'segformer.encoder.'
		state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
		self.load_state_dict(state_dict)

	def forward(self, x):
		features = []

		for embed, blocks, norm in zip(self.patch_embeddings, self.block, self.layer_norm):
			x, h, w = embed(x)

			for block in blocks:
				x = block(x, h, w)

			x = norm(x).transpose(1, 2).reshape(x.size(0), -1, h, w)
			features.append(x)

		return features

import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
import yaml


def load_manifest(path):
	"""Accepts either a manifest .json directly, or a .yaml whose `manifest` key points at one (relative to the yaml)."""
	path = Path(path)
	if path.suffix in ('.yaml', '.yml'):
		with path.open('r') as f:
			cfg = yaml.safe_load(f)
		path = (path.parent / cfg['manifest']).resolve()
	with open(path, 'r') as f:
		return json.load(f)


def resize_filter(src_width, dst_width):
	return Image.BOX if dst_width <= src_width else Image.BICUBIC


def crop_box(width, src_height, dst_height, anchor):
	if anchor == 'top':
		top = 0
	elif anchor == 'bottom':
		top = src_height - dst_height
	else:
		top = (src_height - dst_height) // 2

	return (0, top, width, top + dst_height)


def one_hot_mask(class_ids):
	"""3-class one-hot (0=obstacle, 1=water, 2=sky). Any other value (255 void) maps to all-zero, which every loss and metric treats as ignore."""
	return np.stack([class_ids == 0, class_ids == 1, class_ids == 2], axis=-1).astype(np.float32)


class LaRSDataset(torch.utils.data.Dataset):
	"""LaRS wrapper driven by a resolution-bucket manifest.

	Each __getitem__ takes a (index, (width, height)) pair produced by
	ResolutionBatchSampler and returns the sample downscaled to that size, so
	every batch is internally uniform while resolution varies across batches.
	"""
	def __init__(self, manifest_path, transform=None, normalize_t=None, include_original=False):
		manifest = load_manifest(manifest_path)
		self.samples = manifest['samples']
		self.buckets = [tuple(b) for b in manifest['buckets']]
		self.crop_anchor = manifest.get('crop_anchor', 'center')
		self.transform = transform
		self.normalize_t = normalize_t
		self.include_original = include_original

	def __len__(self):
		return len(self.samples)

	def sample_sizes(self):
		"""Per-sample candidate bucket sizes, consumed by the batch sampler."""
		return [[tuple(s) for s in sample['sizes']] for sample in self.samples]

	def __getitem__(self, index_and_size):
		idx, size = index_and_size
		width, height = size
		sample = self.samples[idx]

		resize_width, resize_height = sample.get('resize', (width, height))

		img = Image.open(sample['image']).convert('RGB')
		mask = Image.open(sample['mask'])

		img = img.resize((resize_width, resize_height), resize_filter(img.size[0], resize_width))
		mask = mask.resize((resize_width, resize_height), Image.NEAREST)

		if (resize_width, resize_height) != (width, height):
			box = crop_box(width, resize_height, height, self.crop_anchor)
			img = img.crop(box)
			mask = mask.crop(box)

		img = np.array(img)
		img_original = img
		mask = one_hot_mask(np.array(mask))

		data = {'image': img, 'segmentation': mask}

		if self.transform is not None:
			data = self.transform(data)
			img = data['image']

		if self.normalize_t is not None:
			img = self.normalize_t(img)
		else:
			img = TF.to_tensor(img)

		features = {'image': img}
		labels = {'segmentation': torch.from_numpy(data['segmentation'].transpose(2, 0, 1))}

		if self.include_original:
			features['image_original'] = torch.from_numpy(img_original.transpose(2, 0, 1))

		labels['img_name'] = Path(sample['image']).stem
		labels['mask_filename'] = Path(sample['mask']).name

		return features, labels


class ResolutionBatchSampler(torch.utils.data.Sampler):
	def __init__(self, sample_sizes, batch_size, train=True, seed=0):
		self.sample_sizes = sample_sizes
		self.batch_size = batch_size
		self.train = train
		self.seed = seed if seed is not None else 0
		self.epoch = 0

	def _world_rank(self):
		if dist.is_available() and dist.is_initialized():
			return dist.get_world_size(), dist.get_rank()
		return 1, 0

	def _num_train_batches(self):
		world, _ = self._world_rank()
		buckets = {size for sizes in self.sample_sizes for size in sizes}
		return (len(self.sample_sizes) // self.batch_size - len(buckets)) // world

	def _train_batches(self, seed):
		rng = random.Random(seed)

		by_bucket = {}
		for idx, sizes in enumerate(self.sample_sizes):
			by_bucket.setdefault(rng.choice(sizes), []).append(idx)

		batches = []
		for size, indices in by_bucket.items():
			rng.shuffle(indices)
			for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
				batches.append([(i, size) for i in indices[start:start + self.batch_size]])

		rng.shuffle(batches)

		world, rank = self._world_rank()
		return batches[:self._num_train_batches() * world][rank::world]

	def _eval_batches(self):
		by_bucket = {}
		for idx, sizes in enumerate(self.sample_sizes):
			for size in sizes:
				by_bucket.setdefault(size, []).append(idx)

		batches = []
		for size, indices in by_bucket.items():
			for start in range(0, len(indices), self.batch_size):
				batches.append([(i, size) for i in indices[start:start + self.batch_size]])

		return batches

	def __iter__(self):
		if not self.train:
			return iter(self._eval_batches())

		batches = self._train_batches(self.seed + self.epoch)
		self.epoch += 1
		return iter(batches)

	def __len__(self):
		if not self.train:
			return len(self._eval_batches())

		return self._num_train_batches()

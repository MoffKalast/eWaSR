import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
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

		img = Image.open(sample['image']).convert('RGB').resize((width, height), Image.BOX)
		mask = Image.open(sample['mask']).resize((width, height), Image.NEAREST)

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
	"""Yields batches of (index, size) pairs, one resolution per batch.

	Every epoch each sample is assigned one of its candidate buckets at random
	(stochastic multiplicity of 1, re-rolled per epoch). Because bigger buckets
	receive more samples, batch counts are naturally proportional to image count.
	The trailing partial batch of each bucket is dropped.
	"""
	def __init__(self, sample_sizes, batch_size, shuffle=True, drop_last=True, seed=0):
		self.sample_sizes = sample_sizes
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.seed = seed
		self.epoch = 0
		self._length = len(self._build_batches(self.seed))

	def _build_batches(self, seed):
		rng = random.Random(seed)

		by_bucket = {}
		for idx, sizes in enumerate(self.sample_sizes):
			chosen = sizes[rng.randrange(len(sizes))] if len(sizes) > 1 else sizes[0]
			by_bucket.setdefault(chosen, []).append(idx)

		batches = []
		for size, indices in by_bucket.items():
			if self.shuffle:
				rng.shuffle(indices)
			limit = len(indices) - (len(indices) % self.batch_size) if self.drop_last else len(indices)
			for start in range(0, limit, self.batch_size):
				chunk = indices[start:start + self.batch_size]
				batches.append([(i, size) for i in chunk])

		if self.shuffle:
			rng.shuffle(batches)
		return batches

	def __iter__(self):
		batches = self._build_batches(self.seed + self.epoch)
		self.epoch += 1
		return iter(batches)

	def __len__(self):
		return self._length

import argparse
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image


DOWNSCALE_FACTORS = (1, 2, 4, 8)


def read_image_list(path):
	with open(path, 'r') as file:
		return [line.strip() for line in file if line.strip()]


def candidate_sizes(width, height, min_width, max_width):
	"""Integer downscales of a native size whose width lands in [min_width, max_width]."""
	sizes = []
	for f in DOWNSCALE_FACTORS:
		w, h = width // f, height // f
		if min_width <= w <= max_width:
			sizes.append((w, h))
	return sizes


def build_manifest(split_dirs, image_subdir, mask_subdir, mask_suffix, image_ext, min_width, max_width, batch_size):
	samples = []
	bucket_counts = defaultdict(int)
	missing = 0

	for split_dir in split_dirs:
		split_dir = Path(split_dir)
		names = read_image_list(split_dir / 'image_list.txt')
		image_dir = split_dir / image_subdir
		mask_dir = split_dir / mask_subdir

		for name in names:
			img_path = image_dir / f'{name}{image_ext}'
			mask_path = mask_dir / f'{name}{mask_suffix}.png'
			if not img_path.exists() or not mask_path.exists():
				missing += 1
				continue

			with Image.open(img_path) as im:
				w, h = im.size

			sizes = candidate_sizes(w, h, min_width, max_width)
			if not sizes:
				continue

			for s in sizes:
				bucket_counts[s] += 1

			samples.append({'image': str(img_path.resolve()), 'mask': str(mask_path.resolve()), 'native': [w, h], 'sizes': sizes})

	# A bucket is only usable if it can fill at least one batch. Images left with no
	# usable bucket are dropped entirely rather than padded.
	kept_buckets = {s for s, c in bucket_counts.items() if c >= batch_size}

	usable_samples = []
	dropped = 0
	for sample in samples:
		sizes = [s for s in sample['sizes'] if tuple(s) in kept_buckets]
		if not sizes:
			dropped += 1
			continue
		sample['sizes'] = sizes
		usable_samples.append(sample)

	manifest = {
		'buckets': sorted([list(b) for b in kept_buckets], key=lambda x: (-x[0], -x[1])),
		'samples': usable_samples
	}

	stats = {
		'missing_files': missing,
		'dropped_no_bucket': dropped,
		'usable_samples': len(usable_samples),
		'bucket_counts': {f'{w}x{h}': bucket_counts[(w, h)] for (w, h) in sorted(kept_buckets, key=lambda x: (-x[0], -x[1]))}
	}
	return manifest, stats


def main():
	parser = argparse.ArgumentParser(description='Build a resolution-bucket manifest for variable-resolution LaRS training.')
	parser.add_argument('--splits', type=str, nargs='+', required=True, help='One or more LaRS split directories (each with image_list.txt, images/, semantic_masks/). Multiple are merged, e.g. train + test.')
	parser.add_argument('--output', type=str, required=True, help='Output manifest JSON path.')
	parser.add_argument('--image_subdir', type=str, default='images')
	parser.add_argument('--mask_subdir', type=str, default='semantic_masks')
	parser.add_argument('--mask_suffix', type=str, default='', help='Suffix appended to the basename before .png for masks (LaRS uses none, MaSTr uses "m").')
	parser.add_argument('--image_ext', type=str, default='.jpg')
	parser.add_argument('--min_width', type=int, default=256)
	parser.add_argument('--max_width', type=int, default=640)
	parser.add_argument('--batch_size', type=int, default=4, help='Buckets with fewer images than this are dropped.')
	args = parser.parse_args()

	manifest, stats = build_manifest(args.splits, args.image_subdir, args.mask_subdir, args.mask_suffix, args.image_ext, args.min_width, args.max_width, args.batch_size)

	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	with open(args.output, 'w') as f:
		json.dump(manifest, f)

	print(f'Wrote {args.output}')
	print(f'  usable samples: {stats["usable_samples"]}')
	print(f'  dropped (no usable bucket): {stats["dropped_no_bucket"]}')
	print(f'  missing files: {stats["missing_files"]}')
	print(f'  buckets ({len(manifest["buckets"])}):')
	for k, c in stats['bucket_counts'].items():
		print(f'    {k:>9}  {c} imgs')


if __name__ == '__main__':
	main()

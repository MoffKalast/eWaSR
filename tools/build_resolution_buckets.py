import argparse
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image


DOWNSCALE_FACTORS = (1, 2, 4, 8)

DEFAULT_MIN_WIDTH = 256
DEFAULT_MAX_WIDTH = 640


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


def fixed_width_plan(width, height, target_width, height_multiple):
	exact_height = max(1, round(height * target_width / width))
	bucket_height = (exact_height // height_multiple) * height_multiple

	if bucket_height < 1:
		return None

	return (target_width, exact_height), (target_width, bucket_height)


def build_manifest(split_dirs, image_subdir, mask_subdir, mask_suffix, image_ext, batch_size, mode, min_width, max_width, target_width, height_multiple):
	samples = []
	bucket_counts = defaultdict(int)
	missing = 0
	cropped = 0
	max_crop = 0

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

			resize = None

			if mode == 'fixed_width':
				plan = fixed_width_plan(w, h, target_width, height_multiple)
				if plan is None:
					continue

				resize_size, bucket = plan
				sizes = [bucket]

				if resize_size != bucket:
					resize = list(resize_size)
					cropped += 1
					max_crop = max(max_crop, resize_size[1] - bucket[1])
			else:
				sizes = candidate_sizes(w, h, min_width, max_width)

			if not sizes:
				continue

			for s in sizes:
				bucket_counts[s] += 1

			sample = {'image': str(img_path.resolve()), 'mask': str(mask_path.resolve()), 'native': [w, h], 'sizes': sizes}
			if resize is not None:
				sample['resize'] = resize

			samples.append(sample)

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
		'cropped_samples': cropped,
		'max_crop_rows': max_crop,
		'bucket_counts': {f'{w}x{h}': bucket_counts[(w, h)] for (w, h) in sorted(kept_buckets, key=lambda x: (-x[0], -x[1]))}
	}
	return manifest, stats


def main():
	parser = argparse.ArgumentParser(description='Build a resolution-bucket manifest for variable-resolution LaRS training.')
	parser.add_argument('--splits', type=str, nargs='+', required=True, help='One or more LaRS split directories (each with image_list.txt, images/, semantic_masks/). Multiple are merged, e.g. train + test.')
	parser.add_argument('--output', type=str, required=True, help='Output manifest JSON path.')
	parser.add_argument('--mode', type=str, default='downscale', choices=['downscale', 'fixed_width'], help='downscale: integer 1/2/4/8 downscales with width in [min_width, max_width]. fixed_width: rescale every image to --width preserving aspect, then bucket by height.')
	parser.add_argument('--image_subdir', type=str, default='images')
	parser.add_argument('--mask_subdir', type=str, default='semantic_masks')
	parser.add_argument('--mask_suffix', type=str, default='', help='Suffix appended to the basename before .png for masks (LaRS uses none, MaSTr uses "m").')
	parser.add_argument('--image_ext', type=str, default='.jpg')
	parser.add_argument('--min_width', type=int, default=None, help=f'downscale mode only. Default {DEFAULT_MIN_WIDTH}.')
	parser.add_argument('--max_width', type=int, default=None, help=f'downscale mode only. Default {DEFAULT_MAX_WIDTH}.')
	parser.add_argument('--width', type=int, default=None, help='fixed_width mode only. Target width every image is rescaled to.')
	parser.add_argument('--height_multiple', type=int, default=8, help='fixed_width mode only. Aspect-preserving height is floored to a multiple of this and the excess rows are cropped, so near-identical aspect ratios share a bucket. Use 1 to disable cropping.')
	parser.add_argument('--crop_anchor', type=str, default='center', choices=['top', 'center', 'bottom'], help='fixed_width mode only. Which part of the frame is kept when rows are cropped.')
	parser.add_argument('--batch_size', type=int, default=4, help='Buckets with fewer images than this are dropped.')
	args = parser.parse_args()

	if args.mode == 'fixed_width':
		if args.width is None:
			parser.error('--width is required in fixed_width mode')
		if args.min_width is not None or args.max_width is not None:
			parser.error('--min_width/--max_width apply to downscale mode only')
		if args.height_multiple < 1:
			parser.error('--height_multiple must be at least 1')
	else:
		if args.width is not None:
			parser.error('--width applies to fixed_width mode only')

	min_width = DEFAULT_MIN_WIDTH if args.min_width is None else args.min_width
	max_width = DEFAULT_MAX_WIDTH if args.max_width is None else args.max_width

	manifest, stats = build_manifest(args.splits, args.image_subdir, args.mask_subdir, args.mask_suffix, args.image_ext, args.batch_size, args.mode, min_width, max_width, args.width, args.height_multiple)

	if args.mode == 'fixed_width':
		manifest['crop_anchor'] = args.crop_anchor

	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	with open(args.output, 'w') as f:
		json.dump(manifest, f)

	print(f'Wrote {args.output}')
	print(f'  mode: {args.mode}')
	print(f'  usable samples: {stats["usable_samples"]}')
	print(f'  dropped (no usable bucket): {stats["dropped_no_bucket"]}')
	print(f'  missing files: {stats["missing_files"]}')
	if args.mode == 'fixed_width':
		print(f'  cropped samples: {stats["cropped_samples"]} (at most {stats["max_crop_rows"]} rows, anchor={args.crop_anchor})')
	print(f'  buckets ({len(manifest["buckets"])}):')
	for k, c in stats['bucket_counts'].items():
		print(f'    {k:>9}  {c} imgs')


if __name__ == '__main__':
	main()

import argparse
import csv
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wasr.models as models
from wasr.utils import load_weights
from datasets.lars import LaRSDataset
from datasets.transforms import PytorchHubNormalization

OBSTACLE, WATER, SKY, IGNORE = 0, 1, 2, 3
PALETTE = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164], [0, 0, 0]], dtype=np.uint8)
METRICS = {'miou': True, 'iou_obstacle': True, 'iou_water': True, 'iou_sky': True, 'accuracy': True, 'water_edge_f1': True, 'false_obstacle': False, 'missed_obstacle': False, 'spurious_blobs': False}
CAPTION_METRICS = ('miou', 'iou_obstacle', 'water_edge_f1', 'false_obstacle', 'missed_obstacle', 'spurious_blobs')

device = None


def get_arguments():
	parser = argparse.ArgumentParser(description='Evaluate the best.pth of every training run on the test set.')
	parser.add_argument('--logs_dir', type=str, default='output/logs')
	parser.add_argument('--test_config', type=str, default='configs/lars_test.yaml')
	parser.add_argument('--output_dir', type=str, default='output/eval')
	parser.add_argument('--runs', type=str, nargs='*', default=None, help='Run names to evaluate (directory names under logs_dir). All runs if omitted.')
	parser.add_argument('--worst', type=int, default=8, help='Number of worst examples saved per run.')
	parser.add_argument('--worst_by', type=str, default='miou', choices=list(METRICS))
	parser.add_argument('--tolerance', type=int, default=2, help='Pixel distance within which a predicted water edge counts as matching the ground truth one.')
	parser.add_argument('--workers', type=int, default=2, help='Data loading workers per evaluation process.')
	parser.add_argument('--batch', type=int, default=1, help='Test images per forward pass; images are batched with others of the same resolution.')
	parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to spread the runs over.')
	parser.add_argument('--parallel_per_gpu', type=int, default=1, help='Runs evaluated at the same time on each GPU.')
	return parser.parse_args()


def find_runs(logs_dir, names):
	runs = []
	for weights in sorted(Path(logs_dir).glob('*/version_*/best.pth')):
		name = weights.parent.parent.name
		if names and name not in names:
			continue

		label = name if weights.parent.name == 'version_0' else '%s_%s' % (name, weights.parent.name)
		runs.append((label, weights))

	return runs


def load_model(weights):
	with open(weights.parent / 'hparams.yaml', 'r') as f:
		hp = yaml.load(f, Loader=yaml.FullLoader)

	model = models.get_model(hp['model'], num_classes=hp['num_classes'], pretrained=False, mixer=hp['mixer'], enricher=hp['enricher'], project=hp['project'], pyramid=hp.get('pyramid', False), sim_interp=hp.get('sim_interp'))
	model.load_state_dict(load_weights(weights))
	return model.eval().to(device)


@torch.no_grad()
def predict(model, image):
	logits = model({'image': image.to(device)})['out']
	logits = F.interpolate(logits, size=image.shape[2:], mode='bilinear', align_corners=False)
	return logits.argmax(1).cpu().numpy()


def hard_labels(segmentation):
	segmentation = segmentation.numpy()
	labels = segmentation.argmax(0)
	labels[segmentation.sum(0) < 0.5] = IGNORE
	return labels.astype(np.uint8)


def water_edge(water, usable):
	return water & ndimage.binary_dilation(~water) & usable


def water_edge_f1(pred, gt, tolerance):
	usable = ~ndimage.binary_dilation(gt == IGNORE, iterations=tolerance + 1)
	pred_edge = water_edge(pred == WATER, usable)
	gt_edge = water_edge(gt == WATER, usable)

	if not pred_edge.any() and not gt_edge.any():
		return np.nan
	if not pred_edge.any() or not gt_edge.any():
		return 0.0

	precision = (ndimage.distance_transform_edt(~gt_edge)[pred_edge] <= tolerance).mean()
	recall = (ndimage.distance_transform_edt(~pred_edge)[gt_edge] <= tolerance).mean()

	if precision + recall == 0:
		return 0.0

	return 2 * precision * recall / (precision + recall)


def spurious_blobs(pred, gt):
	blobs, count = ndimage.label(pred == OBSTACLE)
	touching = np.unique(blobs[gt != WATER])
	return count - np.count_nonzero(touching)


def fraction(values):
	return values.mean() if values.size > 0 else np.nan


def image_metrics(pred, gt, tolerance):
	valid = gt != IGNORE
	result = {}

	for c, name in ((OBSTACLE, 'obstacle'), (WATER, 'water'), (SKY, 'sky')):
		p = (pred == c) & valid
		g = gt == c
		union = np.count_nonzero(p | g)
		result['iou_' + name] = np.count_nonzero(p & g) / union if union > 0 else np.nan

	result['miou'] = np.nanmean([result['iou_obstacle'], result['iou_water'], result['iou_sky']])
	result['accuracy'] = fraction(pred[valid] == gt[valid])
	result['water_edge_f1'] = water_edge_f1(pred, gt, tolerance)
	result['false_obstacle'] = fraction(pred[gt == WATER] == OBSTACLE)
	result['missed_obstacle'] = fraction(pred[gt == OBSTACLE] == WATER)
	result['spurious_blobs'] = spurious_blobs(pred, gt)

	return result


class EvalBatches(Dataset):
	def __init__(self, dataset, items, batches):
		self.dataset = dataset
		self.items = items
		self.batches = batches

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, i):
		samples = [self.dataset[self.items[position]] for position in self.batches[i]]
		image = torch.stack([features['image'] for features, _ in samples])
		gt = torch.from_numpy(np.stack([hard_labels(labels['segmentation']) for _, labels in samples]))
		names = [labels['img_name'] for _, labels in samples]
		return self.batches[i], image, gt, names


def make_batches(items, batch):
	order = sorted(range(len(items)), key=lambda position: tuple(items[position][1]))

	batches = []
	for _, group in itertools.groupby(order, key=lambda position: tuple(items[position][1])):
		group = list(group)
		batches += [group[i:i + batch] for i in range(0, len(group), batch)]

	return batches


def evaluate(model, dataset, items, args, show_progress):
	batches = make_batches(items, args.batch)
	loader = DataLoader(EvalBatches(dataset, items, batches), batch_size=None, num_workers=args.workers, prefetch_factor=1 if args.workers > 0 else None)

	rows = []
	for positions, image, gt, names in tqdm(loader, leave=False, disable=not show_progress):
		pred = predict(model, image)
		for position, p, g, name in zip(positions, pred, gt, names):
			rows.append({'image': name, 'position': position, 'width': g.shape[1], 'height': g.shape[0], **image_metrics(p, g.numpy().astype(np.int64), args.tolerance)})

	return sorted(rows, key=lambda row: row['position'])


def make_panel(image, gt, pred, caption):
	error = (image * 0.4).astype(np.uint8)
	error[(pred != gt) & (gt != IGNORE)] = (255, 40, 40)

	grid = np.concatenate([np.concatenate([image, PALETTE[gt]], axis=1), np.concatenate([PALETTE[pred], error], axis=1)], axis=0)

	canvas = Image.new('RGB', (grid.shape[1], grid.shape[0] + 24))
	canvas.paste(Image.fromarray(grid), (0, 24))
	ImageDraw.Draw(canvas).text((6, 6), caption, fill=(255, 255, 255))

	return canvas


def save_worst(model, dataset, items, rows, run_dir, count, metric):
	ranked = sorted((row for row in rows if not np.isnan(row[metric])), key=lambda row: row[metric], reverse=not METRICS[metric])

	for rank, row in enumerate(ranked[:count], 1):
		features, labels = dataset[items[row['position']]]
		image = features['image_original'].numpy().transpose(1, 2, 0)
		pred = predict(model, features['image'].unsqueeze(0))[0]
		gt = hard_labels(labels['segmentation']).astype(np.int64)

		caption = '%s   ' % row['image'] + '   '.join('%s %.4g' % (m, row[m]) for m in CAPTION_METRICS)
		make_panel(image, gt, pred, caption).save(run_dir / ('worst_%02d_%s.png' % (rank, row['image'])))


def write_csv(path, rows):
	with open(path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def plot_ranking(summary, path):
	height = max(2.5, 0.35 * len(summary) + 1)
	fig, axes = plt.subplots(3, 3, figsize=(20, 3 * height))

	for ax, (metric, higher) in zip(axes.flat, METRICS.items()):
		ranked = sorted(summary, key=lambda row: row[metric], reverse=not higher)
		values = [row[metric] for row in ranked]

		ax.barh([row['label'] for row in ranked], values, color=['tab:green' if i == len(ranked) - 1 else 'tab:blue' for i in range(len(ranked))])
		ax.set_title('%s (%s is better)' % (metric, 'higher' if higher else 'lower'))

		low, high = min(values), max(values)
		pad = (high - low) * 0.5 or abs(high) * 0.05 or 1.0
		ax.set_xlim(max(0.0, low - pad), high + pad)

		for i, value in enumerate(values):
			ax.text(value, i, ' %.4f' % value, va='center', fontsize=8)

	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)


def init_worker(devices):
	global device
	device = torch.device(devices.get())


def evaluate_run(label, weights, args, show_progress):
	print('evaluating %s on %s' % (label, device), flush=True)

	dataset = LaRSDataset(args.test_config, normalize_t=PytorchHubNormalization(), include_original=True)
	items = [(i, max(sizes, key=lambda s: s[0] * s[1])) for i, sizes in enumerate(dataset.sample_sizes())]

	model = load_model(weights)
	rows = evaluate(model, dataset, items, args, show_progress)

	run_dir = Path(args.output_dir) / label
	run_dir.mkdir(parents=True, exist_ok=True)
	write_csv(run_dir / 'per_image.csv', rows)
	save_worst(model, dataset, items, rows, run_dir, args.worst, args.worst_by)

	summary = {'label': label, 'images': len(rows), **{m: float(np.nanmean([row[m] for row in rows])) for m in METRICS}}
	print('%s done on %d images\n  %s' % (label, len(rows), '  '.join('%s %.4f' % (m, summary[m]) for m in METRICS)), flush=True)

	return summary


def main():
	args = get_arguments()

	runs = find_runs(args.logs_dir, args.runs)
	if not runs:
		raise SystemExit('no best.pth found under %s' % args.logs_dir)

	gpus = ['cuda:%d' % i for i in range(args.num_gpu)] if torch.cuda.is_available() else ['cpu']
	slots = min(len(gpus) * args.parallel_per_gpu, len(runs))

	context = multiprocessing.get_context('spawn')
	devices = context.Queue()
	for _ in range(args.parallel_per_gpu):
		for gpu in gpus:
			devices.put(gpu)

	print('evaluating %d runs, %d at a time over %s' % (len(runs), slots, ', '.join(gpus)), flush=True)

	with ProcessPoolExecutor(max_workers=slots, mp_context=context, initializer=init_worker, initargs=(devices,)) as pool:
		futures = [pool.submit(evaluate_run, label, weights, args, slots == 1) for label, weights in runs]
		summary = [future.result() for future in futures]

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	write_csv(output_dir / 'summary.csv', sorted(summary, key=lambda row: -row['miou']))
	plot_ranking(summary, output_dir / 'ranking.png')
	print('wrote %s and %s' % (output_dir / 'summary.csv', output_dir / 'ranking.png'))


if __name__ == '__main__':
	main()

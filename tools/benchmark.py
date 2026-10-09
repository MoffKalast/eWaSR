import sys
import os

sys.path.append(os.getcwd())

import argparse
import torch

from wasr.mit import VARIANTS
from wasr.models import EWaSR

def get_arguments():
	parser = argparse.ArgumentParser(description='Measure PyTorch inference latency on random weights and inputs.')
	parser.add_argument('--variant', type=str, choices=list(VARIANTS), default='b0')
	parser.add_argument('--width', type=int, default=640)
	parser.add_argument('--height', type=int, default=192)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--ch_sim', type=int, default=256)
	parser.add_argument('--mixer', type=str, default='CCCCSS')
	parser.add_argument('--enricher', type=str, default='SS')
	parser.add_argument('--fp16', action='store_true', help='Cast the model and input to half precision.')
	parser.add_argument('--warmup', type=int, default=20)
	parser.add_argument('--iterations', type=int, default=200)
	return parser.parse_args()

def benchmark(args):
	dtype = torch.float16 if args.fp16 else torch.float32
	model = EWaSR(args.variant, pretrain=None, ch_sim=args.ch_sim, mixer=args.mixer, enricher=args.enricher).eval().to('cuda', dtype)
	x = torch.randn(args.batch_size, 3, args.height, args.width, device='cuda', dtype=dtype)

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	with torch.inference_mode():
		for _ in range(args.warmup):
			model(x)

		torch.cuda.synchronize()
		start.record()

		for _ in range(args.iterations):
			model(x)

		end.record()
		torch.cuda.synchronize()

	latency = start.elapsed_time(end) / args.iterations
	encoder = sum(p.numel() for p in model.encoder.parameters()) / 1e6
	decoder = sum(p.numel() for p in model.decoder.parameters()) / 1e6

	print(f'{args.variant} {args.width}x{args.height} batch {args.batch_size} {"fp16" if args.fp16 else "fp32"}')
	print(f'  params: encoder {encoder:.2f}M, decoder {decoder:.2f}M')
	print(f'  latency: {latency:.2f} ms per batch, {1000 * args.batch_size / latency:.1f} images/s')

def main():
	benchmark(get_arguments())

if __name__ == '__main__':
	main()

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets.transforms import PytorchHubNormalization
from wasr.mit import VARIANTS
from wasr.models import EWaSR
from wasr.utils import load_weights

SEGMENTATION_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def get_arguments():
	parser = argparse.ArgumentParser(description='Run eWaSR on a folder of images and save color-coded overlays.')
	parser.add_argument('--images', type=str, required=True, help='Folder with input images.')
	parser.add_argument('--weights', type=str, required=True, help='Path to the model weights or a training checkpoint.')
	parser.add_argument('--output_dir', type=str, required=True, help='Folder the overlays are written to.')
	parser.add_argument('--variant', type=str, choices=list(VARIANTS), default='b0')
	parser.add_argument('--width', type=int, default=640, help='Images are resized to this width, keeping their aspect ratio.')
	parser.add_argument('--ch_sim', type=int, default=256)
	parser.add_argument('--mixer', type=str, default='CCCCSS')
	parser.add_argument('--enricher', type=str, default='SS')
	parser.add_argument('--fp16', action='store_true', help='Run in half precision.')
	return parser.parse_args()

def predict(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.float16 if args.fp16 else torch.float32

	model = EWaSR(args.variant, pretrain=None, ch_sim=args.ch_sim, mixer=args.mixer, enricher=args.enricher)
	model.load_state_dict(load_weights(args.weights))
	model = model.eval().to(device, dtype)

	normalize = PytorchHubNormalization()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

	for path in tqdm(paths):
		img = Image.open(path).convert('RGB')
		height = round(img.height * args.width / img.width)
		img = np.array(img.resize((args.width, height), Image.BILINEAR))

		with torch.inference_mode():
			logits = model(normalize(img).unsqueeze(0).to(device, dtype))['out']
			logits = F.interpolate(logits.float(), size=img.shape[:2], mode='bilinear', align_corners=False)

		classes = logits.argmax(1)[0].cpu().numpy()
		overlay = (img * 0.7 + SEGMENTATION_COLORS[classes] * 0.3).astype(np.uint8)
		Image.fromarray(overlay).save(output_dir / f'{path.stem}.png')

def main():
	predict(get_arguments())

if __name__ == '__main__':
	main()

# eWaSR - an embedded-compute-ready maritime obstacle detection network

Segformer backbone version of [eWaSR](https://github.com/tersekmatija/eWaSR).

## Setup

**Requirements**: Python >= 3.8, PyTorch >= 2.0, PyTorch Lightning >= 2.0

```bash
pip install -r requirements.txt
```

## Model

The encoder is the Mix Transformer (MiT) from SegFormer, in the `b0` or `b1` size, followed by the eWaSR decoder. The encoder can be initialized from ImageNet classification (`--pretrain imagenet`) or from the encoder of SegFormer finetuned for semantic segmentation on ADE20K (`--pretrain ade`, default) or Cityscapes (`--pretrain cityscapes`). Weights are downloaded from the official `nvidia/*` HuggingFace repositories on first use.

## Training data

Training uses LaRS through a resolution manifest built by `tools/build_resolution_buckets.py`. Two modes are available:

* `downscale`: every image is used at its native size divided by 1, 2, 4 or 8, keeping only sizes with a width inside `[--min_width, --max_width]`.
* `fixed_width`: every image is rescaled to `--width` keeping its aspect ratio, so batches only differ in height. Heights are floored to a multiple of `--height_multiple` by cropping a few rows.

```bash
python tools/build_resolution_buckets.py --splits /data/LaRS/train --output configs/lars_train_manifest.json --mode fixed_width --width 640
python tools/build_resolution_buckets.py --splits /data/LaRS/val --output configs/lars_val_manifest.json --mode fixed_width --width 640
```

## Model training

```bash
python train.py \
--train_config configs/lars_train.yaml \
--val_config configs/lars_val.yaml \
--model_name ewasr_b0 \
--variant b0 \
--pretrain ade \
--batch_size 8 \
--epochs 100
```

Training uses AdamW with a linear warmup followed by polynomial decay, and the decoder gets a 10x higher learning rate than the encoder. Logs and exported weights (`best.pth`, `weights.pth`) are stored in `output/logs/<model_name>`.

```bash
tensorboard --logdir output/logs/<model_name>
```

## Inference

```bash
python predict.py \
--images examples/images \
--weights output/logs/ewasr_b0/version_0/best.pth \
--variant b0 \
--output_dir output/predictions
```

Predictions are stored as color-coded overlays.

`tools/benchmark.py` measures PyTorch latency for a given variant, resolution and batch size on random weights.


## License

This repository, including pre-trained weights, is licensed under Apache-2.0.

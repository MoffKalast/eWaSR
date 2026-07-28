import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

def water_obstacle_separation_loss(features, gt_mask, isolation_kernel=9, isolation_power=2.0, clip=100.0, min_samples=0.0):
    """Computes the water-obstacle separation loss from intermediate features.

    Minimizes, per channel, the ratio of the within-water feature variance to the mean squared
    distance of obstacle features from the water mean. Water is required to be compact while
    obstacles are only required to be far, so an isolated few-pixel buoy is rewarded for being an
    outlier rather than penalized for it. Reflections are labelled water, which is what pulls their
    features onto the water cluster.

    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor
        isolation_kernel (int): Neighbourhood size for obstacle isolation weighting, 1 to disable
        isolation_power (float): Exponent on the inverse local density. At 1.0 an isolated pixel in a
            9x9 window is upweighted ~81x, which is still small against the ~3000x pixel-count
            advantage a shoreline has over a buoy. Raise it to push harder toward small obstacles.
            This is the one number here worth sweeping.
        clip (float): Per-channel ceiling on the ratio, the loss is unbounded without it
        min_samples (float): Minimum per-image water and obstacle coverage, in feature cells, for the
            image to contribute. Now per-image rather than batch-wide, so the old value of 5 is far
            stricter than it looks: at 1/16 stride an 8-pixel buoy is ~0.03 cells, and gating those
            images out defeats the isolation weighting below. Defaults to 0, since the epsilon in the
            denominator and the clip already cover the numerical stability this used to provide.

    Returns:
        (loss, skipped_fraction) where skipped_fraction is the share of the batch that failed
        min_samples and therefore contributed nothing.
    """
    epsilon_watercost = 0.01

    # Resize gt mask to match the extracted features shape (x,y)
    feature_size = (features.size(2), features.size(3))
    gt_mask = F.interpolate(gt_mask, size=feature_size, mode='area')

    # (1 = water, 2 = sky, 0 = obstacles)
    mask_water = gt_mask[:, 1:2]
    mask_obstacles = gt_mask[:, 0:1]

    # All statistics below are per-image and reduced over the batch only at the very end, so a single
    # frame with a large shoreline cannot set the water mean for every other frame in the batch.
    count_water = mask_water.sum((2, 3), keepdim=True)
    count_obstacles = mask_obstacles.sum((2, 3), keepdim=True)

    valid = (count_water > min_samples) & (count_obstacles > min_samples)
    num_valid = valid.sum()

    # An obstacle pixel counts for more when few of its neighbours are also obstacles. Without this
    # the obstacle term is an area-weighted mean dominated by the interior of large shore and pier
    # regions, and a 4-pixel buoy contributes ~0. Stands in for inverse-instance-area weighting,
    # which would need the panoptic labels that the semantic PNG loader does not read.
    weight_obstacles = mask_obstacles
    if isolation_kernel > 1:
        density = F.avg_pool2d(mask_obstacles, isolation_kernel, stride=1, padding=isolation_kernel // 2)
        weight_obstacles = mask_obstacles / (density + 1.0 / (isolation_kernel ** 2)).pow(isolation_power)

    norm_water = count_water.clamp(min=1.0)
    norm_obstacles = weight_obstacles.sum((2, 3), keepdim=True).clamp(min=1e-6)

    mean_water = (mask_water * features).sum((2, 3), keepdim=True) / norm_water

    sq_dist = (features - mean_water).pow(2)
    var_water = (mask_water * sq_dist).sum((2, 3), keepdim=True) / norm_water
    dist_obstacles = (weight_obstacles * sq_dist).sum((2, 3), keepdim=True) / norm_obstacles

    loss_c = (var_water / (dist_obstacles + epsilon_watercost)).clamp(max=clip)

    # Mean over channels, then over the valid images
    loss = (loss_c.mean(1, keepdim=True) * valid).sum() / num_valid.clamp(min=1)
    skipped = 1.0 - num_valid.item() / gt_mask.size(0)

    return loss, skipped

def focal_loss(logits, labels, gamma=2.0, alpha=4.0, target_scale='labels'):
    """Focal loss of the segmentation output `logits` and ground truth `labels`."""

    epsilon = 1.e-9

    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        labels = F.interpolate(labels, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=InterpolationMode.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)

    # Focal loss
    fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
    fl = fl.sum(1) # Sum focal loss along channel dimension

    # Return mean of the focal loss along spatial and batch dimensions
    return fl.mean()

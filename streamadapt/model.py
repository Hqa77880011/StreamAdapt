import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VideoModel(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        layers, inputs = [], 3
        for channels in (width // 2, width, width):
            layers.extend([nn.Conv2d(inputs, channels, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU()])
            inputs = channels
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(width, classes, 1)
        self.predictor = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, frames):
        features = self.encoder(frames)
        dense = self.classifier(features)
        return dense.mean((0, 2, 3)), features, dense


def image_tensor(frames, device):
    return torch.as_tensor(np.stack(frames), device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255


def jsd(first, second, dimension=1):
    midpoint = (first + second) / 2
    return 0.5 * ((first * ((first + 1e-8).log() - (midpoint + 1e-8).log())).sum(dimension)
                  + (second * ((second + 1e-8).log() - (midpoint + 1e-8).log())).sum(dimension))


def motion_grid(previous, current, size, device, align=True):
    height, width = size
    ys, xs = np.mgrid[:height, :width].astype(np.float32)
    if align:
        first = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        second = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (width, height))
        flow[:, :, 0] *= width / current.shape[1]
        flow[:, :, 1] *= height / current.shape[0]
        xs, ys = xs + flow[:, :, 0], ys + flow[:, :, 1]
    valid = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    grid = np.stack([2 * xs / max(width - 1, 1) - 1, 2 * ys / max(height - 1, 1) - 1], -1)
    previous_small = cv2.resize(previous, (width, height)).astype(np.float32) / 255
    current_small = cv2.resize(current, (width, height)).astype(np.float32) / 255
    warped = cv2.remap(previous_small, xs, ys, cv2.INTER_LINEAR)
    stable = valid & (np.abs(warped - current_small).mean(2) < 0.1)
    return (torch.as_tensor(grid, device=device)[None], torch.as_tensor(valid, device=device, dtype=torch.float32)[None],
            torch.as_tensor(stable, device=device, dtype=torch.float32)[None])


def prequential(model, previous, current, args, index, entropy_only=False, no_motion=False):
    model.eval()
    frames = image_tensor([previous, current], args.device)
    _, features, dense = model(frames)
    probability = dense.softmax(1)
    entropy = -(probability[1:] * probability[1:].clamp_min(1e-8).log()).sum(1).mean()
    if entropy_only:
        return entropy
    grid, valid, stable = motion_grid(previous, current, features.shape[-2:], args.device, not no_motion)
    warped = F.grid_sample(probability[:1], grid, align_corners=True, padding_mode="zeros")
    warped = warped / warped.sum(1, keepdim=True).clamp_min(1e-8)
    consistency = (jsd(warped, probability[1:]) * valid).sum() / valid.sum().clamp_min(1)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + index)
    hidden = (torch.rand((1, 1, *features.shape[-2:]), generator=generator, device=args.device) < 0.25).float()
    recovered = model.predictor(features[1:] * (1 - hidden))
    feature_error = ((recovered - features[1:].detach()).square() * hidden).sum() / (hidden.sum() * features.shape[1]).clamp_min(1)
    old_confidence = warped.max(1).values
    current_confidence = probability[1:].max(1).values
    track_loss = (F.relu(old_confidence - current_confidence) * stable).sum() / stable.sum().clamp_min(1)
    return args.motion_weight * consistency + args.feature_weight * feature_error + args.track_weight * track_loss


def adapt(model, frames, args, index):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adaptable = []
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if args.method == "tent":
                layer.track_running_stats = False
                layer.running_mean = None
                layer.running_var = None
            layer.weight.requires_grad_(True)
            layer.bias.requires_grad_(True)
            adaptable.extend([layer.weight, layer.bias])
    optimizer = torch.optim.Adam(adaptable, lr=args.adapt_lr)
    model.eval()
    for _ in range(args.adapt_steps):
        _, _, dense = model(image_tensor(frames, args.device))
        probability = dense.softmax(1)
        objective = -(probability * probability.clamp_min(1e-8).log()).sum(1).mean()
        if len(frames) >= 2 and args.method != "tent":
            objective = objective + prequential(model, frames[-2], frames[-1], args, index)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    return model

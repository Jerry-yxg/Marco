import os
import csv
import math
import time
import random
import argparse
import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import registry
from utils_fl import partition_data, DatasetSplit, get_dataset, get_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(path):
    logger = logging.getLogger('fedfca_multiscale_rff')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser('Multi-scale RFF + group regularization + entropy-safe dual-space FedFCA')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--partition', default='dir')
    parser.add_argument('--partition_min_size', type=int, default=10)
    parser.add_argument('--partition_max_retries', type=int, default=100)
    parser.add_argument('--model_name', default='resnet18_imagenet')
    parser.add_argument('--data_root', default='/root/autodl-tmp/FedCGS-main/FedCGS-main/data/')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--feature_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--local_epochs', type=int, default=30)
    parser.add_argument('--head_batch_size', type=int, default=128)
    parser.add_argument('--head_lr', type=float, default=0.05)
    parser.add_argument('--head_momentum', type=float, default=0.9)
    parser.add_argument('--head_weight_decay', type=float, default=0.0001)
    parser.add_argument('--head_prox_mu', type=float, default=0.001)
    parser.add_argument('--curv_batch_size', type=int, default=512)
    parser.add_argument('--curv_temp', type=float, default=2.0)
    parser.add_argument('--curv_kappa', type=float, default=20.0)
    parser.add_argument('--curv_damping', type=float, default=0.0001)
    parser.add_argument('--curv_clip_min', type=float, default=0.01)
    parser.add_argument('--curv_clip_max', type=float, default=20.0)
    parser.add_argument('--rff_bandwidths', nargs='+', type=float, default=[1.0, 2.0])
    parser.add_argument('--rff_dims', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--rff_etas', nargs='+', type=float, default=[0.25, 0.25])
    parser.add_argument('--rff_group_mus', nargs='+', type=float, default=[0.001, 0.001])
    parser.add_argument('--expansion_seed', type=int, default=-1, help='Negative means seed + 1984; seed 42 therefore reproduces expansion seed 2026.')
    parser.add_argument('--rff_eps', type=float, default=1e-12)
    parser.add_argument('--bayes_gamma', type=float, default=0.0)
    parser.add_argument('--cov_jitter', type=float, default=0.0001)
    parser.add_argument('--fusion_weights', nargs=3, type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument('--safety_mode', choices=['smoothstep', 'linear', 'hard'], default='smoothstep')
    parser.add_argument('--entropy_low', type=float, default=0.40,
                        help='At or below this normalized client-label entropy, use pure MSRFF-Bayes.')
    parser.add_argument('--entropy_high', type=float, default=0.60,
                        help='At or above this entropy, retain the original fixed fusion weights.')
    parser.add_argument('--summary_file', default='',
                        help='Optional path. Append exactly one clean FINAL line after each run.')
    parser.add_argument('--baseline_lambda', type=float, default=0.5)
    parser.add_argument('--target_gain_vs_fedcgs', type=float, default=0.036)
    parser.add_argument('--single_rff_reference_acc', type=float, default=0.667, help='Previous best aligned single-scale RFF result, diagnostic only.')
    parser.add_argument('--log_dir', default='./fedfca_multiscale_rff_logs')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    args.datasets = args.dataset
    lengths = {len(args.rff_bandwidths), len(args.rff_dims), len(args.rff_etas), len(args.rff_group_mus)}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError('rff_bandwidths, rff_dims, rff_etas and rff_group_mus must have the same non-zero length')
    if any((value <= 0 for value in args.rff_bandwidths)):
        raise ValueError('Every RFF bandwidth must be positive')
    if any((value <= 0 for value in args.rff_dims)):
        raise ValueError('Every RFF dimension must be positive')
    if any((value <= 0 for value in args.rff_etas)):
        raise ValueError('Every RFF eta must be positive')
    if any((value < 0 for value in args.rff_group_mus)):
        raise ValueError('Every RFF group regularization strength must be non-negative')
    if not (0.0 <= args.entropy_low < args.entropy_high <= 1.0):
        raise ValueError('Require 0 <= entropy_low < entropy_high <= 1')
    weights = np.asarray(args.fusion_weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('fusion_weights must be non-negative and sum to > 0')
    args.fusion_weights = (weights / weights.sum()).tolist()
    args.effective_expansion_seed = args.expansion_seed if args.expansion_seed >= 0 else args.seed + 1984
    return args

@torch.no_grad()
def extract_features(model, loader, device, expected_dim):
    feature_parts, label_parts = ([], [])
    model.eval()
    for images, labels in loader:
        output = model(images.to(device, non_blocking=True), return_features=True)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError('return_features=True must return (logits, features)')
        features = output[1]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if features.ndim != 2 or features.size(1) != expected_dim:
            raise RuntimeError(f'Unexpected feature shape: {tuple(features.shape)}')
        feature_parts.append(features.float().cpu())
        label_parts.append(labels.long().cpu())
    return (torch.cat(feature_parts), torch.cat(label_parts))

@dataclass
class RFFMap:
    bandwidth: float
    dim: int
    eta: float
    omega: torch.Tensor
    phase: torch.Tensor

@torch.no_grad()
def rff_component(features, rff_map, raw_dim, device, batch_size, eps):
    """Gaussian RFF on unit-normalized backbone features."""
    scale = math.sqrt(2.0 * raw_dim / rff_map.dim)
    omega = rff_map.omega.to(device)
    phase = rff_map.phase.to(device)
    outputs = []
    loader = torch.utils.data.DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    for features_cpu, in loader:
        normalized = F.normalize(features_cpu.to(device, non_blocking=True), p=2, dim=1, eps=eps)
        projection = normalized @ omega.T / float(rff_map.bandwidth)
        outputs.append((scale * torch.cos(projection + phase)).cpu())
    return torch.cat(outputs)

@torch.no_grad()
def multiscale_features(raw, rff_maps, device, batch_size, eps):
    blocks = [raw]
    raw_dim = raw.size(1)
    for rff_map in rff_maps:
        component = rff_component(raw, rff_map, raw_dim, device, batch_size, eps)
        blocks.append(float(rff_map.eta) * component)
    return torch.cat(blocks, dim=1)

def build_rff_maps(args, raw_dim):
    generator = torch.Generator().manual_seed(args.effective_expansion_seed)
    maps = []
    for bandwidth, dim, eta in zip(args.rff_bandwidths, args.rff_dims, args.rff_etas):
        maps.append(RFFMap(bandwidth=float(bandwidth), dim=int(dim), eta=float(eta), omega=torch.randn(int(dim), raw_dim, generator=generator), phase=2.0 * math.pi * torch.rand(int(dim), generator=generator)))
    return maps

def expanded_initialization(raw_weight, raw_bias, total_rff_dim):
    num_classes, raw_dim = raw_weight.shape
    weight = torch.zeros(num_classes, raw_dim + total_rff_dim, dtype=torch.float32)
    weight[:, :raw_dim] = raw_weight
    return (weight, raw_bias.clone())

@dataclass
class LocalPackage:
    n: int
    theta: torch.Tensor
    precision: torch.Tensor

def make_loader(features, labels, args, client_id, shuffle):
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(args.seed + 1009 * (client_id + 1))
    return torch.utils.data.DataLoader(TensorDataset(features, labels), batch_size=args.head_batch_size, shuffle=shuffle, generator=generator)

def local_train_accuracy(head, features, labels, args, device):
    head.eval()
    correct = 0
    loader = torch.utils.data.DataLoader(TensorDataset(features, labels), batch_size=args.head_batch_size, shuffle=False)
    with torch.no_grad():
        for features_cpu, labels_cpu in loader:
            prediction = head(features_cpu.to(device)).argmax(1).cpu()
            correct += int(prediction.eq(labels_cpu).sum())
    return correct / max(len(labels), 1)

def train_raw_head(features, labels, init_weight, init_bias, args, device, client_id):
    head = nn.Linear(features.size(1), init_weight.size(0)).to(device)
    with torch.no_grad():
        head.weight.copy_(init_weight.to(device))
        head.bias.copy_(init_bias.to(device))
    optimizer = torch.optim.SGD([{'params': [head.weight], 'weight_decay': args.head_weight_decay}, {'params': [head.bias], 'weight_decay': 0.0}], lr=args.head_lr, momentum=args.head_momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.local_epochs, 1))
    loader = make_loader(features, labels, args, client_id, shuffle=True)
    init_weight_device = init_weight.to(device)
    init_bias_device = init_bias.to(device)
    for _ in range(args.local_epochs):
        head.train()
        for features_cpu, labels_cpu in loader:
            features_device = features_cpu.to(device)
            labels_device = labels_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features_device), labels_device)
            loss += 0.5 * args.head_prox_mu * ((head.weight - init_weight_device).pow(2).sum() + (head.bias - init_bias_device).pow(2).sum())
            loss.backward()
            optimizer.step()
        scheduler.step()
    return (head, local_train_accuracy(head, features, labels, args, device))

def train_multiscale_head(features, labels, init_weight, init_bias, raw_dim, rff_dims, rff_group_mus, args, device, client_id):
    """Raw block anchored to pretrained head; each RFF scale shrunk separately."""
    head = nn.Linear(features.size(1), init_weight.size(0)).to(device)
    with torch.no_grad():
        head.weight.copy_(init_weight.to(device))
        head.bias.copy_(init_bias.to(device))
    optimizer = torch.optim.SGD(head.parameters(), lr=args.head_lr, momentum=args.head_momentum, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.local_epochs, 1))
    loader = make_loader(features, labels, args, client_id, shuffle=True)
    init_raw = init_weight[:, :raw_dim].to(device)
    init_bias_device = init_bias.to(device)
    boundaries = []
    start = raw_dim
    for dim, group_mu in zip(rff_dims, rff_group_mus):
        end = start + int(dim)
        boundaries.append((start, end, float(group_mu)))
        start = end
    for _ in range(args.local_epochs):
        head.train()
        for features_cpu, labels_cpu in loader:
            features_device = features_cpu.to(device)
            labels_device = labels_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_weight = head.weight[:, :raw_dim]
            loss = F.cross_entropy(head(features_device), labels_device)
            loss += 0.5 * args.head_weight_decay * raw_weight.pow(2).sum()
            loss += 0.5 * args.head_prox_mu * ((raw_weight - init_raw).pow(2).sum() + (head.bias - init_bias_device).pow(2).sum())
            for start, end, group_mu in boundaries:
                loss += 0.5 * group_mu * head.weight[:, start:end].pow(2).sum()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return (head, local_train_accuracy(head, features, labels, args, device))

def pack_head(head):
    return torch.cat([head.weight.detach().cpu().double(), head.bias.detach().cpu().double().unsqueeze(1)], dim=1)

@torch.no_grad()
def full_curvature(head, features, class_count, args, device):
    num_classes, feature_dim = (head.out_features, features.size(1))
    curvature = torch.zeros(num_classes, feature_dim + 1, dtype=torch.float64)
    loader = torch.utils.data.DataLoader(TensorDataset(features), batch_size=args.curv_batch_size, shuffle=False)
    for features_cpu, in loader:
        probabilities = torch.softmax(head(features_cpu.to(device)) / args.curv_temp, dim=1)
        uncertainty = (probabilities * (1.0 - probabilities)).double().cpu()
        squared = features_cpu.double().pow(2)
        augmented_squared = torch.cat([squared, torch.ones(squared.size(0), 1, dtype=torch.float64)], dim=1)
        curvature += torch.einsum('bc,br->cr', uncertainty, augmented_squared)
    curvature /= max(len(features), 1)
    reliability = class_count / (class_count + args.curv_kappa)
    precision = curvature * reliability[:, None]
    mask = (class_count > 0)[:, None].expand_as(precision)
    precision = (precision + args.curv_damping) * mask
    active = precision[mask]
    if active.numel() == 0:
        raise RuntimeError('Client has no valid class for curvature aggregation')
    precision /= active.mean() + 1e-12
    return torch.where(mask, precision.clamp(args.curv_clip_min, args.curv_clip_max), torch.zeros_like(precision))

def aggregate_full_curvature(packages):
    sample_weights = torch.tensor([package.n for package in packages], dtype=torch.float64)
    sample_weights /= sample_weights.sum()
    denominator = sum((weight * package.precision for weight, package in zip(sample_weights, packages)))
    numerator = sum((weight * package.precision * package.theta for weight, package in zip(sample_weights, packages)))
    return numerator / (denominator + 1e-12)

def feature_statistics(features, labels, num_classes, batch_size=2048):
    feature_dim = features.size(1)
    count = torch.zeros(num_classes, dtype=torch.float64)
    class_sum = torch.zeros(num_classes, feature_dim, dtype=torch.float64)
    feature_sum = torch.zeros(feature_dim, dtype=torch.float64)
    second_moment = torch.zeros(feature_dim, feature_dim, dtype=torch.float64)
    loader = torch.utils.data.DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    for features_cpu, labels_cpu in loader:
        features_double = features_cpu.double()
        labels_long = labels_cpu.long()
        class_sum.scatter_add_(0, labels_long[:, None].expand(-1, feature_dim), features_double)
        count += torch.bincount(labels_long, minlength=num_classes).double()
        feature_sum += features_double.sum(0)
        second_moment += features_double.T @ features_double
    return (count, class_sum, feature_sum, second_moment)

def add_stats(total, current):
    if total is None:
        return tuple((value.clone() for value in current))
    return tuple((first + second for first, second in zip(total, current)))

def collect_statistics(client_features, client_labels, num_classes):
    total = None
    counts = []
    for features, labels in zip(client_features, client_labels):
        statistics = feature_statistics(features, labels, num_classes)
        counts.append(statistics[0])
        total = add_stats(total, statistics)
    return (total, counts)


def normalized_client_label_entropy(class_counts):
    """Sample-weighted normalized label entropy and label-space coverage."""
    if not class_counts:
        raise ValueError('class_counts must be non-empty')
    num_classes = int(class_counts[0].numel())
    total_samples = float(sum(float(count.sum()) for count in class_counts))
    if total_samples <= 0 or num_classes <= 1:
        return (0.0, 0.0)
    entropy = 0.0
    coverage = 0.0
    normalizer = math.log(float(num_classes))
    for count in class_counts:
        client_n = float(count.sum())
        if client_n <= 0:
            continue
        probabilities = (count / client_n).double()
        positive = probabilities > 0
        client_entropy = float(-(probabilities[positive] * probabilities[positive].log()).sum() / normalizer)
        client_coverage = float(positive.double().mean())
        sample_weight = client_n / total_samples
        entropy += sample_weight * client_entropy
        coverage += sample_weight * client_coverage
    return (float(np.clip(entropy, 0.0, 1.0)), float(np.clip(coverage, 0.0, 1.0)))


def entropy_safety_coefficient(entropy, mode, low, high):
    """Return rho in [0,1]: 0=pure Bayes, 1=original fixed fusion."""
    entropy = float(np.clip(entropy, 0.0, 1.0))
    if mode == 'linear':
        return entropy
    if mode == 'hard':
        return float(entropy >= high)
    t = float(np.clip((entropy - low) / (high - low), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def entropy_safe_weights(base_weights, rho):
    """Shrink both curvature experts together while preserving their 2:3 ratio."""
    raw_weight, expanded_weight, bayes_weight = map(float, base_weights)
    rho = float(np.clip(rho, 0.0, 1.0))
    safe_raw = rho * raw_weight
    safe_expanded = rho * expanded_weight
    safe_bayes = bayes_weight + (1.0 - rho) * (raw_weight + expanded_weight)
    safe = np.asarray([safe_raw, safe_expanded, safe_bayes], dtype=np.float64)
    safe /= safe.sum()
    return safe.tolist()

def stable_cholesky(matrix, base_jitter, max_tries=8):
    identity = torch.eye(matrix.size(0), dtype=torch.float64)
    jitter = 0.0
    for attempt in range(max_tries):
        cholesky, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if int(info.max()) == 0:
            return (cholesky, jitter)
        jitter = max(base_jitter, 1e-12) * 10.0 ** attempt
    raise RuntimeError('Covariance is not positive definite')

def bayes_head(statistics, gamma, base_jitter):
    count, class_sum, feature_sum, second_moment = statistics
    if torch.any(count <= 0):
        missing = torch.where(count <= 0)[0].tolist()
        raise RuntimeError(f'Globally missing classes: {missing}')
    sample_count = float(count.sum())
    means = class_sum / count[:, None]
    global_mean = feature_sum / sample_count
    covariance = (second_moment - sample_count * torch.outer(global_mean, global_mean)) / (sample_count - 1.0)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += gamma * torch.eye(covariance.size(0), dtype=torch.float64)
    cholesky, jitter = stable_cholesky(covariance, base_jitter)
    weight = torch.cholesky_solve(means.T, cholesky).T
    prior = (count / sample_count).clamp_min(1e-12)
    bias = torch.log(prior) - 0.5 * (means * weight).sum(1)
    return (torch.cat([weight, bias[:, None]], dim=1), jitter)

def center_head(theta):
    return theta - theta.mean(dim=0, keepdim=True)

def moment_logit_scale(theta, statistics):
    count, _, feature_sum, second_moment = statistics
    sample_count = float(count.sum())
    feature_dim = feature_sum.numel()
    gram = torch.zeros(feature_dim + 1, feature_dim + 1, dtype=torch.float64)
    gram[:feature_dim, :feature_dim] = second_moment
    gram[:feature_dim, feature_dim] = feature_sum
    gram[feature_dim, :feature_dim] = feature_sum
    gram[feature_dim, feature_dim] = sample_count
    centered = center_head(theta)
    energy = (centered @ gram * centered).sum() / (sample_count * centered.size(0))
    return float(torch.sqrt(energy.clamp_min(1e-12)))

@torch.no_grad()
def head_logits(theta, features, device, batch_size):
    theta_device = center_head(theta).to(device)
    outputs = []
    loader = torch.utils.data.DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    for features_cpu, in loader:
        features_device = features_cpu.double().to(device)
        outputs.append((features_device @ theta_device[:, :-1].T + theta_device[:, -1]).cpu())
    return torch.cat(outputs)

def accuracy(logits, labels):
    return float(logits.argmax(1).eq(labels).double().mean())

def three_expert_metrics(raw_logits, expanded_logits, bayes_logits, fused, labels):
    raw_prediction = raw_logits.argmax(1)
    expanded_prediction = expanded_logits.argmax(1)
    bayes_prediction = bayes_logits.argmax(1)
    raw_correct = raw_prediction.eq(labels)
    expanded_correct = expanded_prediction.eq(labels)
    bayes_correct = bayes_prediction.eq(labels)
    return {'acc_raw_fc': float(raw_correct.double().mean()), 'acc_exp_fc': float(expanded_correct.double().mean()), 'acc_bayes': float(bayes_correct.double().mean()), 'acc_fused': accuracy(fused, labels), 'oracle_three': float((raw_correct | expanded_correct | bayes_correct).double().mean()), 'only_raw_fc': int((raw_correct & ~expanded_correct & ~bayes_correct).sum()), 'only_exp_fc': int((~raw_correct & expanded_correct & ~bayes_correct).sum()), 'only_bayes': int((~raw_correct & ~expanded_correct & bayes_correct).sum()), 'all_wrong': int((~raw_correct & ~expanded_correct & ~bayes_correct).sum())}

def train_curvature_space(client_features, client_labels, class_counts, init_weight, init_bias, args, device, logger, space_name, raw_dim=None):
    packages, train_accuracies = ([], [])
    for client_id, (features, labels, count) in enumerate(zip(client_features, client_labels, class_counts)):
        if raw_dim is None:
            head, train_accuracy = train_raw_head(features, labels, init_weight, init_bias, args, device, client_id)
        else:
            head, train_accuracy = train_multiscale_head(features, labels, init_weight, init_bias, raw_dim, args.rff_dims, args.rff_group_mus, args, device, client_id)
        precision = full_curvature(head, features, count, args, device)
        packages.append(LocalPackage(len(labels), pack_head(head), precision))
        train_accuracies.append(train_accuracy)
        logger.info('[CLIENT-%s] id=%02d n=%d train_acc=%.4f', space_name, client_id, len(labels), train_accuracy)
        del head, precision
    return (aggregate_full_curvature(packages), float(np.mean(train_accuracies)))

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    base_name = f'fedfca_multiscale_rff_{args.dataset}_beta{args.beta:g}_seed{args.seed}{tag}_{timestamp}'
    log_path = os.path.join(args.log_dir, base_name + '.log')
    csv_path = os.path.join(args.log_dir, base_name + '.csv')
    logger = setup_logger(log_path)
    logger.info('===== RUN CONFIG =====')
    for key, value in sorted(vars(args).items()):
        logger.info('%s = %s', key, value)
    logger.info('======================')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    num_classes, train_set, test_set = get_dataset(args.dataset, args.data_root)
    train_set, test_set, groups, _ = partition_data(train_set, test_set, args.partition, beta=args.beta, num_users=args.num_users, logger=logger, args=args)
    model, raw_dim = get_model(args.model_name, num_classes)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, 'fc') or not isinstance(model.fc, nn.Linear):
        raise AttributeError('The model must expose model.fc as nn.Linear')
    raw_init_weight = model.fc.weight.detach().cpu().float().clone()
    raw_init_bias = model.fc.bias.detach().cpu().float().clone()
    raw_clients, client_labels = ([], [])
    for client_id in range(len(groups)):
        loader = torch.utils.data.DataLoader(DatasetSplit(train_set, groups[client_id]), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        features, labels = extract_features(model, loader, device, raw_dim)
        raw_clients.append(features)
        client_labels.append(labels)
        logger.info('[FEATURE] client=%02d n=%d', client_id, len(labels))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    raw_test, test_labels = extract_features(model, test_loader, device, raw_dim)
    raw_statistics, raw_counts = collect_statistics(raw_clients, client_labels, num_classes)
    label_entropy, label_coverage = normalized_client_label_entropy(raw_counts)
    safety_rho = entropy_safety_coefficient(
        label_entropy, args.safety_mode, args.entropy_low, args.entropy_high
    )
    safe_weights = entropy_safe_weights(args.fusion_weights, safety_rho)
    logger.info(
        '[SAFETY] label_entropy=%.4f label_coverage=%.4f mode=%s '
        'thresholds=(%.2f,%.2f) rho=%.4f base_weights=(%.4f,%.4f,%.4f) '
        'safe_weights=(%.4f,%.4f,%.4f)',
        label_entropy, label_coverage, args.safety_mode,
        args.entropy_low, args.entropy_high, safety_rho,
        *args.fusion_weights, *safe_weights
    )
    theta_raw_fc, raw_train_mean = train_curvature_space(raw_clients, client_labels, raw_counts, raw_init_weight, raw_init_bias, args, device, logger, 'RAW-FC')
    theta_raw_bayes, raw_jitter = bayes_head(raw_statistics, args.bayes_gamma, args.cov_jitter)
    raw_fc_logits = head_logits(theta_raw_fc, raw_test, device, args.test_batch_size)
    raw_bayes_logits = head_logits(theta_raw_bayes, raw_test, device, args.test_batch_size)
    baseline_alpha = args.baseline_lambda / (1.0 + args.baseline_lambda)
    baseline_logits = (1.0 - baseline_alpha) * raw_fc_logits + baseline_alpha * raw_bayes_logits
    acc_raw_fc = accuracy(raw_fc_logits, test_labels)
    acc_raw_bayes = accuracy(raw_bayes_logits, test_labels)
    acc_baseline = accuracy(baseline_logits, test_labels)
    target = acc_raw_bayes + args.target_gain_vs_fedcgs
    raw_fc_scale = moment_logit_scale(theta_raw_fc, raw_statistics)
    logger.info('[BASELINE] RawFC=%.4f Bayes=%.4f FedFCA=%.4f target=%.4f raw_train_mean=%.4f scale_raw_fc=%.4f jitter=%.2e', acc_raw_fc, acc_raw_bayes, acc_baseline, target, raw_train_mean, raw_fc_scale, raw_jitter)
    rff_maps = build_rff_maps(args, raw_dim)
    logger.info('[MS-RFF] bands=%s dims=%s etas=%s mus=%s total_rff_dim=%d', args.rff_bandwidths, args.rff_dims, args.rff_etas, args.rff_group_mus, sum(args.rff_dims))
    expanded_clients = [multiscale_features(features, rff_maps, device, args.feature_batch_size, args.rff_eps) for features in raw_clients]
    expanded_test = multiscale_features(raw_test, rff_maps, device, args.feature_batch_size, args.rff_eps)
    expanded_statistics, expanded_counts = collect_statistics(expanded_clients, client_labels, num_classes)
    total_rff_dim = sum(args.rff_dims)
    expanded_init_weight, expanded_init_bias = expanded_initialization(raw_init_weight, raw_init_bias, total_rff_dim)
    theta_ms_fc, expanded_train_mean = train_curvature_space(expanded_clients, client_labels, expanded_counts, expanded_init_weight, expanded_init_bias, args, device, logger, 'MSRFF-FC', raw_dim=raw_dim)
    theta_ms_bayes, expanded_jitter = bayes_head(expanded_statistics, args.bayes_gamma, args.cov_jitter)
    ms_fc_logits = head_logits(theta_ms_fc, expanded_test, device, args.test_batch_size)
    ms_bayes_logits = head_logits(theta_ms_bayes, expanded_test, device, args.test_batch_size)
    ms_fc_scale = moment_logit_scale(theta_ms_fc, expanded_statistics)
    ms_bayes_scale = moment_logit_scale(theta_ms_bayes, expanded_statistics)
    weight_raw, weight_ms_fc, weight_ms_bayes = args.fusion_weights
    safe_raw_weight, safe_ms_fc_weight, safe_bayes_weight = safe_weights

    fixed_fusion_raw = (
        weight_raw * raw_fc_logits
        + weight_ms_fc * ms_fc_logits
        + weight_ms_bayes * ms_bayes_logits
    )
    fixed_fusion_aligned = (
        weight_raw * raw_fc_logits / raw_fc_scale
        + weight_ms_fc * ms_fc_logits / ms_fc_scale
        + weight_ms_bayes * ms_bayes_logits / ms_bayes_scale
    )
    safe_fusion_raw = (
        safe_raw_weight * raw_fc_logits
        + safe_ms_fc_weight * ms_fc_logits
        + safe_bayes_weight * ms_bayes_logits
    )
    safe_fusion_aligned = (
        safe_raw_weight * raw_fc_logits / raw_fc_scale
        + safe_ms_fc_weight * ms_fc_logits / ms_fc_scale
        + safe_bayes_weight * ms_bayes_logits / ms_bayes_scale
    )
    metrics = three_expert_metrics(
        raw_fc_logits, ms_fc_logits, ms_bayes_logits, safe_fusion_aligned, test_labels
    )
    acc_fixed_raw = accuracy(fixed_fusion_raw, test_labels)
    acc_fixed_aligned = accuracy(fixed_fusion_aligned, test_labels)
    acc_safe_raw = accuracy(safe_fusion_raw, test_labels)
    gain_vs_fedcgs = metrics['acc_fused'] - acc_raw_bayes
    gain_vs_fedfca = metrics['acc_fused'] - acc_baseline
    gain_vs_single_rff = metrics['acc_fused'] - args.single_rff_reference_acc
    block_norms = []
    block_start = raw_dim
    for bandwidth, block_dim in zip(args.rff_bandwidths, args.rff_dims):
        block_end = block_start + block_dim
        block_norms.append((float(bandwidth), float(torch.linalg.vector_norm(theta_ms_fc[:, block_start:block_end]).item())))
        block_start = block_end
    raw_block_norm = float(torch.linalg.vector_norm(theta_ms_fc[:, :raw_dim]).item())
    logger.info('[RESULT] dataset=%s beta=%.4g seed=%d D=%d RawFC=%.4f MSRFF-FC=%.4f MSRFF-Bayes=%.4f FixedRaw=%.4f FixedAligned=%.4f SafeRaw=%.4f SafeAligned=%.4f Oracle3=%.4f only_raw=%d only_msrff=%d only_bayes=%d all_wrong=%d train_msrff_mean=%.4f label_entropy=%.4f label_coverage=%.4f safety_rho=%.4f raw_norm=%.4f rff_block_norms=%s scales=(%.4f,%.4f,%.4f) base_weights=(%.2f,%.2f,%.2f) safe_weights=(%.4f,%.4f,%.4f) gain_vs_SingleRFF=%+.4f gain_vs_FedCGS=%+.4f gain_vs_FedFCA=%+.4f target=%.4f reached=%d jitter=%.2e', args.dataset, args.beta, args.seed, raw_dim + total_rff_dim, metrics['acc_raw_fc'], metrics['acc_exp_fc'], metrics['acc_bayes'], acc_fixed_raw, acc_fixed_aligned, acc_safe_raw, metrics['acc_fused'], metrics['oracle_three'], metrics['only_raw_fc'], metrics['only_exp_fc'], metrics['only_bayes'], metrics['all_wrong'], expanded_train_mean, label_entropy, label_coverage, safety_rho, raw_block_norm, block_norms, raw_fc_scale, ms_fc_scale, ms_bayes_scale, weight_raw, weight_ms_fc, weight_ms_bayes, safe_raw_weight, safe_ms_fc_weight, safe_bayes_weight, gain_vs_single_rff, gain_vs_fedcgs, gain_vs_fedfca, target, int(metrics['acc_fused'] >= target), expanded_jitter)
    row = {'dataset': args.dataset, 'beta': args.beta, 'seed': args.seed, 'expansion_seed': args.effective_expansion_seed, 'rff_bandwidths': '|'.join(map(str, args.rff_bandwidths)), 'rff_dims': '|'.join(map(str, args.rff_dims)), 'rff_etas': '|'.join(map(str, args.rff_etas)), 'rff_group_mus': '|'.join(map(str, args.rff_group_mus)), 'total_dim': raw_dim + total_rff_dim, 'label_entropy': label_entropy, 'label_coverage': label_coverage, 'safety_mode': args.safety_mode, 'entropy_low': args.entropy_low, 'entropy_high': args.entropy_high, 'safety_rho': safety_rho, 'base_weight_raw': weight_raw, 'base_weight_msrff_fc': weight_ms_fc, 'base_weight_bayes': weight_ms_bayes, 'safe_weight_raw': safe_raw_weight, 'safe_weight_msrff_fc': safe_ms_fc_weight, 'safe_weight_bayes': safe_bayes_weight, 'acc_raw_fc': metrics['acc_raw_fc'], 'acc_raw_bayes': acc_raw_bayes, 'acc_original_fedfca': acc_baseline, 'acc_msrff_fc': metrics['acc_exp_fc'], 'acc_msrff_bayes': metrics['acc_bayes'], 'acc_fixed_fusion_raw': acc_fixed_raw, 'acc_fixed_fusion_aligned': acc_fixed_aligned, 'acc_safe_fusion_raw': acc_safe_raw, 'acc_safe_fusion_aligned': metrics['acc_fused'], 'oracle_three': metrics['oracle_three'], 'only_raw_fc': metrics['only_raw_fc'], 'only_msrff_fc': metrics['only_exp_fc'], 'only_msrff_bayes': metrics['only_bayes'], 'all_wrong': metrics['all_wrong'], 'raw_train_mean': raw_train_mean, 'msrff_train_mean': expanded_train_mean, 'scale_raw_fc': raw_fc_scale, 'scale_msrff_fc': ms_fc_scale, 'scale_msrff_bayes': ms_bayes_scale, 'gain_vs_single_rff': gain_vs_single_rff, 'gain_vs_fedcgs': gain_vs_fedcgs, 'gain_vs_fedfca': gain_vs_fedfca, 'target_acc': target, 'target_reached': int(metrics['acc_fused'] >= target), 'jitter': expanded_jitter}
    for index, (bandwidth, norm) in enumerate(block_norms):
        row[f'rff_block_{index}_bandwidth'] = bandwidth
        row[f'rff_block_{index}_norm'] = norm
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    final_line = (
        f"FINAL | dataset={args.dataset} | beta={args.beta:g} | seed={args.seed} "
        f"| H={label_entropy:.3f} | rho={safety_rho:.3f} "
        f"| acc={metrics['acc_fused'] * 100:.2f}% "
        f"| MSRFF-Bayes={metrics['acc_bayes'] * 100:.2f}% "
        f"| FedCGS={acc_raw_bayes * 100:.2f}% "
        f"| gain={gain_vs_fedcgs * 100:+.2f}pp"
    )
    logger.info('[FINAL] %s', final_line)
    logger.info('All experiments finished.\n  log: %s\n  csv: %s', log_path, csv_path)
    print(final_line)
    if args.summary_file:
        summary_path = os.path.abspath(args.summary_file)
        summary_dir = os.path.dirname(summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(summary_path, 'a', encoding='utf-8') as summary:
            summary.write(final_line + '\n')
if __name__ == '__main__':
    main()
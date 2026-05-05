# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sample_data')
parser.add_argument('--method', type=str, default='LDCBR')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--missing_rate', type=float, default=0.5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--force_retrain', action='store_true', help='是否强制重新训练，忽略已保存的模型')
args = parser.parse_args()

try:
    from utils.metrics import evaluation_KLD, evaluation_lt
except Exception as e:
    raise ImportError("未找到 utils.metrics，确保项目结构正确且包含 evaluation_KLD / evaluation_lt。") from e


def _fallback_set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


try:
    from utils.utils import set_seed
except Exception:
    set_seed = _fallback_set_seed

from methods.LDCBR import LDCBR

set_seed(args.seed)

if args.device.lower().startswith('cuda') and not torch.cuda.is_available():
    print("[Warn] CUDA 不可用，自动切换到 CPU。")
    args.device = 'cpu'
device = torch.device(args.device)

data_root = r'E:\LCDBR\dataset'
data_dir = os.path.join(data_root, args.dataset)

feature_npy_path = os.path.join(data_dir, 'feature.npy')
label_npy_path = os.path.join(data_dir, 'label.npy')

X = None
Y = None

if os.path.isfile(feature_npy_path) and os.path.isfile(label_npy_path):
    print(f"[Info] 使用 .npy 数据格式加载: {feature_npy_path}, {label_npy_path}")
    X = np.load(feature_npy_path)
    Y = np.load(label_npy_path)
else:
    mat_file_path = os.path.join(data_root, f'{args.dataset}.mat')
    if not os.path.isfile(mat_file_path):
        raise FileNotFoundError(
            f"未找到 .npy 或 .mat 数据文件：\n"
            f"  - {feature_npy_path}\n"
            f"  - {label_npy_path}\n"
            f"  - {mat_file_path}"
        )
    print(f"[Info] 未找到 .npy，回退使用 .mat 文件: {mat_file_path}")
    import scipy.io
    mat_data = scipy.io.loadmat(mat_file_path)

    feat_keys = ['features', 'X', 'data', 'feature', 'Inputs']
    label_keys = ['labels', 'Y', 'label', 'Targets']

    def _find_key(d, candidates):
        for k in candidates:
            if k in d:
                return k
        raise KeyError(f"在 .mat 中未找到任一候选键：{candidates}")

    X_key = _find_key(mat_data, feat_keys)
    Y_key = _find_key(mat_data, label_keys)

    X = mat_data[X_key]
    Y = mat_data[Y_key]

X = X.astype(np.float32)
Y = Y.astype(np.float32)
if Y.ndim == 1:
    Y = Y[:, None]

print(f"[Info] 特征形状 X: {X.shape}, 标签形状 Y: {Y.shape}")


def apply_missing_labels(y, missing_rate=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    y_missing = y.copy()
    keep_mask = rng.random(y.shape) >= missing_rate
    y_missing[~keep_mask] = 0.0
    return y_missing


def get_model_by_name(method, x_train, y_train, args):
    if method == 'LDCBR':
        model = LDCBR(
            num_feature=x_train.shape[1],
            num_classes=y_train.shape[1],
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            device=args.device,
        )
        return model
    else:
        raise ValueError(f"Unknown method: {method}")


def train_fold(model, train_loader, val_loader, max_epoch, train_path):
    os.makedirs(train_path, exist_ok=True)

    best_state_dict = None
    best_optimizer_state_dict = None
    best_epoch = 0
    min_result = np.inf

    for epoch in range(max_epoch):
        model.train()
        total_train_loss = 0.0
        batch_count = 0

        for x_batch, y_missing_batch, y_full_batch in train_loader:
            x_batch = x_batch.to(model.device, dtype=torch.float32)
            y_missing_batch = y_missing_batch.to(model.device, dtype=torch.float32)
            y_full_batch = y_full_batch.to(model.device, dtype=torch.float32)

            train_loss = model.train_step(x_batch, y_missing_batch, y_full_batch)
            total_train_loss += float(train_loss)
            batch_count += 1

        avg_train_loss = total_train_loss / max(1, batch_count)
        print(f'Epoch {epoch + 1}/{max_epoch}, Training Loss: {avg_train_loss:.6f}')

        preds, ys = model.get_result(val_loader)
        val_kld = evaluation_KLD(ys, preds)

        if val_kld < min_result:
            min_result = val_kld
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(model.optimizer.state_dict())
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': best_optimizer_state_dict,
            }, os.path.join(train_path, 'best.tar'))

    print(f"[Info] Best Val KLD: {min_result:.6f} @ epoch {best_epoch + 1}")
    return min_result


def test_fold(model, train_path, test_loader):
    best_model_path = os.path.join(train_path, "best.tar")
    if hasattr(model, "load"):
        if os.path.exists(best_model_path):
            model.load(best_model_path)
        else:
            print(f"[Warn] 未找到模型文件 {best_model_path}，将使用当前随机权重进行测试！")

    preds, ys = model.get_result(test_loader)
    return preds


kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
metric_names = ['Chebyshev', 'Clark', 'Canberra', 'KLD', 'Cosine', 'Intersection']
metric_results = {name: [] for name in metric_names}
folds_time = []

rng = np.random.default_rng(args.seed)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f'\n=== Fold {fold + 1}/10 | Method: {args.method} | Dataset: {args.dataset} ===')
    print(f'Missing rate (target): {args.missing_rate * 100:.1f}%')

    x_train_all, y_train_all = X[train_idx], Y[train_idx]
    x_test, y_test = X[test_idx], Y[test_idx]

    y_train_all_missing = apply_missing_labels(
        y_train_all, missing_rate=args.missing_rate, rng=rng
    )

    y_test_missing = y_test.copy()

    actual_missing_rate = (y_train_all_missing == 0).mean()

    inner_kf = KFold(n_splits=9, shuffle=True, random_state=args.seed)
    ti, vi = list(inner_kf.split(x_train_all))[0]

    x_train, y_train = x_train_all[ti], y_train_all[ti]
    x_val, y_val = x_train_all[vi], y_train_all[vi]

    y_train_missing_fold = y_train_all_missing[ti]
    y_val_missing = y_train_all_missing[vi]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train_missing_fold).float(),
        torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val_missing).float(),
        torch.from_numpy(y_val).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test_missing).float(),
        torch.from_numpy(y_test).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    train_path = os.path.join('save', 'lt', f'{args.method}', f'fold_{fold}', f'{args.dataset}')
    os.makedirs(train_path, exist_ok=True)
    best_model_file = os.path.join(train_path, 'best.tar')

    model = get_model_by_name(args.method, x_train, y_train, args)
    model.to(args.device)

    t0 = time.time()

    if os.path.exists(best_model_file) and not args.force_retrain:
        print(f"[Info] 检测到已训练模型: {best_model_file}")
        print(f"[Info] 跳过训练阶段，直接加载模型进行测试...")
        fold_time = 0.0
    else:
        if args.force_retrain:
            print("[Info] 强制重训模式已开启...")
        print("[Info] 开始训练...")
        train_fold(model, train_loader, val_loader, args.max_epoch, train_path)
        fold_time = time.time() - t0

    folds_time.append(fold_time)

    y_pred = test_fold(model, train_path, test_loader)

    result = evaluation_lt(y_test, y_pred, y_train=y_train_all)
    for name in metric_names:
        metric_results[name].append(result[name])

    print(f'Fold {fold + 1} results:')
    for name in metric_names:
        print(f'{name}: {result[name]:.4f}')

print('\n=== 10-Fold Results ===')
print(f'Missing rate (target): {args.missing_rate * 100:.1f}%')

for name in metric_names:
    arr = np.array(metric_results[name])
    mean = arr.mean()
    std = arr.std()
    vals = ', '.join([f'{v:.4f}' for v in arr])
    print(f'{name}: {vals}\n Mean: {mean:.4f}, Std: {std:.4f}')

print(f"\nPer-fold time (s): {', '.join([f'{t:.2f}' for t in folds_time])}")
print(f"Avg time (s): {np.mean(folds_time):.2f}")

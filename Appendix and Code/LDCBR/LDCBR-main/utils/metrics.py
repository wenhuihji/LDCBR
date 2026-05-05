import numpy as np

def positive(x):
    return np.maximum(x, 1e-14)

def chebyshev(true, pred):
    return np.mean(np.max(np.abs(true - pred), axis=1))

def clark(true, pred):
    return np.mean(np.sqrt(np.sum((true - pred) ** 2 / positive((true + pred) ** 2), axis=1)))

def canberra(true, pred):
    return np.mean(np.sum(np.abs(true - pred) / positive((true + pred)), axis=1))

def kld(true, pred):
    return np.mean(np.sum(true * np.log(positive(true / positive(pred))), axis=1))

def cosine(true, pred):
    return np.mean(np.sum(true * pred, axis=1) /
                   positive(np.sum(true ** 2, axis=1) ** 0.5) /
                   positive(np.sum(pred ** 2, axis=1) ** 0.5))

def intersection(true, pred):
    return np.mean(np.sum(np.minimum(true, pred), axis=1))

def euclidean(true, pred):
    return np.mean(np.sum((true - pred) ** 2, axis=1) ** 0.5)

def auto_define_head_tail(y_train, topk_ratio=0.5):
    """
    自动根据标签频率划分 head / tail 类别
    参数:
        y_train: [N, C] 标签分布矩阵
        topk_ratio: head 类别占比 (默认 0.5 表示前 50% 标签为 head)
    返回:
        tail_indices, head_indices: list[int]
    """
    if isinstance(y_train, str):
        raise TypeError("y_train 参数应为 numpy.ndarray 类型，而非字符串。请传入训练标签矩阵。")

    label_freq = y_train.sum(axis=0)
    sorted_indices = np.argsort(-label_freq)  # 从高到低排序
    topk = int(len(label_freq) * topk_ratio)
    head = sorted_indices[:topk]
    tail = sorted_indices[topk:]
    return list(tail), list(head)

def evaluation_lt(true, pred, *, y_train):
    """
    计算多个评价指标（包括 head / tail 误差）
    参数:
        true: 测试集真实标签分布 [N, C]
        pred: 模型预测标签分布 [N, C]
        y_train: 训练集标签分布 [N_train, C]，用于自动划分 tail/head
    返回:
        字典形式的指标结果
    """
    tail, head = auto_define_head_tail(y_train)
    tail, head = np.array(tail), np.array(head)

    Chebyshev = chebyshev(true, pred)
    Clark = clark(true, pred)
    Canberra = canberra(true, pred)
    KLD = kld(true, pred)
    Cosine = cosine(true, pred)
    Intersection = intersection(true, pred)

    all_dist = euclidean(true, pred)
    tail_dist = euclidean(true[:, tail], pred[:, tail])
    head_dist = euclidean(true[:, head], pred[:, head])

    return {
        'Chebyshev': Chebyshev,
        'Clark': Clark,
        'Canberra': Canberra,
        'KLD': KLD,
        'Cosine': Cosine,
        'Intersection': Intersection,
        'all': all_dist,
        'tail': tail_dist,
        'head': head_dist
    }

def evaluation_KLD(true, pred):
    return kld(true, pred)

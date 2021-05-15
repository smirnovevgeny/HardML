from math import log2

from torch import Tensor, sort

def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    _, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices].numpy()
    print(sorted_true)
    wrong_sorted = 0
    for i, x in enumerate(sorted_true[:-1]):
        for y in sorted_true[i:]:
            wrong_sorted += int(x < y)
    return wrong_sorted


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "const":
        return y_value
    elif gain_scheme == "exp2":
        return 2 ** y_value - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices].numpy()
    gain = compute_gain(sorted_true, gain_scheme)
    discount = [log2(float(x)) for x in range(2, ys_true.size()[0] + 2)]
    discounted_gain = float((gain / discount).sum())
    return discounted_gain


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    current_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return current_dcg / ideal_dcg



def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    """При реализации precission_at_k необходимо добиться того,
    что максимум функции в единице был достижим при любом ys_true, 
    за исключением не содержащего единиц.
    """
    if not ys_true.sum().bool():
        return -1.0
 
    k = min(k, ys_true.size()[0])
    _, indices = sort(ys_pred, descending=True)
    subset = ys_true[indices][:k]
    k = min(ys_true.sum().item(), k)
    return subset.sum().item() / k


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices]
    return float(1 / (sorted_true.argmax() + 1))


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    sorted_preds, indices = sort(ys_pred, descending=True)
    sorted_true = ys_true[indices]
    break_path = Tensor([(1 - p_break) ** i for i in range(1, ys_true.size()[0])])
    x = ((1 - sorted_true[:-1]).cumprod(dim=0) * break_path * sorted_true[1:]).sum()
    return float(x + sorted_true[0])

def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    n_positive = int(ys_true.sum())
    if not n_positive:
        return -1.0
    else:
        _, indices = sort(ys_pred, descending=True)
        sorted_true = ys_true[indices]
        tp = sorted_true.cumsum(0)
        pos = Tensor(list(range(1, tp.size()[0] + 1))).float()
        precision = tp.div(pos)
        ap = float(precision[sorted_true.bool()].sum() / n_positive)
        return ap
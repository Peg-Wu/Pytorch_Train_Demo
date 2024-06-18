import math
import torch
import pickle

def save_to_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)

def load_from_pickle(load_path):
    with open(load_path, "rb") as f:
        obj = pickle.load(f)
    return obj

class Accumulator():
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@torch.no_grad()
def calc_accuracy(logits, targets, Normalize=True):
    # logits and targets should be torch.Tensor
    assert all(isinstance(var, torch.Tensor) for var in (logits, targets))

    # logits: (batch_size, num_classes)
    outputs = logits.argmax(dim=-1).long()
    targets = targets.reshape(outputs.shape).long()
    
    # pred num_true
    num_true = torch.eq(outputs, targets).sum().item()

    if Normalize:
        return num_true / len(targets)
    else:
        return num_true

@torch.no_grad()
def calc_tp_tn_fp_fn(logits, targets):
    # logits and targets should be torch.Tensor
    assert all(isinstance(var, torch.Tensor) for var in (logits, targets))

    # logits: (batch_size, num_classes)
    outputs = logits.argmax(dim=-1).long()
    targets = targets.reshape(outputs.shape).long()

    # calculate TP, TN, FP, and FN
    TP = ((outputs == 1) & (targets == 1)).sum().item()
    TN = ((outputs == 0) & (targets == 0)).sum().item()
    FP = ((outputs == 1) & (targets == 0)).sum().item()
    FN = ((outputs == 0) & (targets == 1)).sum().item()

    return [TP, TN, FP, FN]

class Two_Clf_Metrics:
    """
    - Metrics for two classification tasks.
    
    - examples:
        >>> import torch
        >>> logits = torch.tensor([[0.1, 0.9],
                                   [0.2, 0.8],
                                   [0.6, 0.4],
                                   [0.7, 0.3]])
        >>> targets = torch.tensor([1, 0, 0, 1])
        >>> metrics = Two_Clf_Metrics(logits=logits, targets=targets)
        >>> print(metrics())

        outputs:
                {'accuracy': 0.49999987500003124,
                 'mcc': 0.0,
                 'sn/recall': 0.499999750000125,
                 'sp': 0.499999750000125,
                 'precision': 0.499999750000125,
                 'f1': 0.49999987500003124}     
    """
    
    def __init__(self, logits, targets, eps=1e-6):
        self.TP, self.TN, self.FP, self.FN = calc_tp_tn_fp_fn(logits, targets)
        self.eps = eps

    @property
    def accuracy(self):
        return (self.TP + self.TN) / \
               (self.TP + self.TN + self.FP + self.FN + self.eps)
    @property
    def mcc(self):
        return (self.TP * self.TN - self.FP * self.FN) / \
               math.sqrt((self.TP + self.FP + self.eps) * (self.TP + self.FN + self.eps) *
                         (self.TN + self.FP + self.eps) * (self.TN + self.FN + self.eps))
    @property
    def sn(self):
        """recall"""
        return self.TP / (self.TP + self.FN + self.eps)
    
    @property
    def sp(self):
        return self.TN / (self.TN + self.FP + self.eps)
    
    @property
    def precision(self):
        return self.TP / (self.TP + self.FP + self.eps)
    
    @property
    def f1(self):
        return (2 * self.TP) / (2 * self.TP + self.FP + self.FN + self.eps)
    
    def __call__(self):
        return {
            "accuracy": self.accuracy,
            "mcc": self.mcc,
            "sn/recall": self.sn,
            "sp": self.sp,
            "precision": self.precision,
            "f1": self.f1
        }
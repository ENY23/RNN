import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

Array = np.ndarray


def _kaiming(shape: Tuple[int, ...], fan_in: int, rng: np.random.Generator) -> Array:
    scale = np.sqrt(2.0 / max(1, fan_in))
    return rng.standard_normal(shape, dtype=np.float32) * scale


def _zeros(shape: Tuple[int, ...]) -> Array:
    return np.zeros(shape, dtype=np.float32)


def _im2col_indices(x: Array, field_height: int, field_width: int,
                    padding: int, stride: int) -> Array:
    n, c, h, w = x.shape
    out_height = (h + 2 * padding - field_height) // stride + 1
    out_width = (w + 2 * padding - field_width) // stride + 1

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant"
    )

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * c)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), field_height * field_width).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * c, -1)
    return cols.astype(np.float32, copy=False)


def _col2im_indices(cols: Array, x_shape: Tuple[int, ...],
                    field_height: int, field_width: int,
                    padding: int, stride: int) -> Array:
    n, c, h, w = x_shape
    h_padded, w_padded = h + 2 * padding, w + 2 * padding
    out_height = (h + 2 * padding - field_height) // stride + 1
    out_width = (w + 2 * padding - field_width) // stride + 1

    cols_reshaped = cols.reshape(c, field_height, field_width, n, out_height, out_width)
    cols_reshaped = cols_reshaped.transpose(3, 0, 4, 5, 1, 2)

    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=np.float32)
    for i in range(field_height):
        i_end = i + stride * out_height
        for j in range(field_width):
            j_end = j + stride * out_width
            x_padded[:, :, i:i_end:stride, j:j_end:stride] += cols_reshaped[:, :, :, :, i, j]

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


@dataclass
class ConvCache:
    x_shape: Tuple[int, int, int, int]
    w: Array
    stride: int
    padding: int
    x_cols: Array
    out_shape: Tuple[int, int, int, int]


@dataclass
class PoolCache:
    x_shape: Tuple[int, int, int, int]
    pool_size: int
    stride: int
    x_cols: Array
    max_idx: Array


@dataclass
class RNNCache:
    x_seq: Array
    h_states: Array
    Wx: Array
    Wh: Array


@dataclass
class LinearCache:
    x: Array
    w: Array


def _conv_forward(x: Array, w: Array, b: Array, stride: int, padding: int,
                  retain_cache: bool) -> Tuple[Array, Optional[ConvCache]]:
    n, _, h, ww = x.shape
    f, _, kh, kw = w.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (ww + 2 * padding - kw) // stride + 1

    x_cols = _im2col_indices(x, kh, kw, padding, stride)
    w_cols = w.reshape(f, -1)
    out = w_cols @ x_cols + b.reshape(-1, 1)
    out = out.reshape(f, out_h, out_w, n).transpose(3, 0, 1, 2)

    cache = None
    if retain_cache:
        cache = ConvCache(
            x_shape=x.shape,
            w=w,
            stride=stride,
            padding=padding,
            x_cols=x_cols,
            out_shape=(n, f, out_h, out_w)
        )
    return out, cache


def _conv_backward(dout: Array, cache: ConvCache) -> Tuple[Array, Array, Array]:
    w = cache.w
    stride = cache.stride
    padding = cache.padding
    x_cols = cache.x_cols
    n, f, out_h, out_w = cache.out_shape
    kh, kw = w.shape[2], w.shape[3]

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(f, -1)
    dw = (dout_reshaped @ x_cols.T).reshape(w.shape)
    db = dout_reshaped.sum(axis=1)
    w_cols = w.reshape(f, -1)
    dx_cols = w_cols.T @ dout_reshaped
    dx = _col2im_indices(dx_cols, cache.x_shape, kh, kw, padding, stride)
    return dx, dw, db


def _relu_forward(x: Array, retain_cache: bool) -> Tuple[Array, Optional[Array]]:
    out = np.maximum(0.0, x)
    cache = x if retain_cache else None
    return out, cache


def _relu_backward(dout: Array, cache: Array) -> Array:
    dx = dout.copy()
    dx[cache <= 0.0] = 0.0
    return dx


def _maxpool_forward(x: Array, pool_size: int, stride: int,
                      retain_cache: bool) -> Tuple[Array, Optional[PoolCache]]:
    n, c, h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    x_reshaped = x.reshape(n * c, 1, h, w)
    x_cols = _im2col_indices(x_reshaped, pool_size, pool_size, 0, stride)
    max_idx = np.argmax(x_cols, axis=0)
    out = x_cols[max_idx, np.arange(max_idx.size)]
    out = out.reshape(out_h, out_w, n, c).transpose(2, 3, 0, 1)

    cache = None
    if retain_cache:
        cache = PoolCache(
            x_shape=x.shape,
            pool_size=pool_size,
            stride=stride,
            x_cols=x_cols,
            max_idx=max_idx
        )
    return out, cache


def _maxpool_backward(dout: Array, cache: PoolCache) -> Array:
    pool_size = cache.pool_size
    stride = cache.stride
    x_shape = cache.x_shape
    n, c, h, w = x_shape

    dcols = np.zeros_like(cache.x_cols)
    dout_flat = dout.transpose(2, 3, 0, 1).ravel()
    dcols[cache.max_idx, np.arange(cache.max_idx.size)] = dout_flat
    dx = _col2im_indices(
        dcols,
        (n * c, 1, h, w),
        pool_size,
        pool_size,
        0,
        stride
    )
    return dx.reshape(x_shape)


def _linear_forward(x: Array, w: Array, b: Array,
                    retain_cache: bool) -> Tuple[Array, Optional[LinearCache]]:
    out = x @ w.T + b
    cache = LinearCache(x=x, w=w) if retain_cache else None
    return out, cache


def _linear_backward(dout: Array, cache: LinearCache) -> Tuple[Array, Array, Array]:
    dx = dout @ cache.w
    dw = dout.T @ cache.x
    db = dout.sum(axis=0)
    return dx, dw, db


def _rnn_forward(seq: Array, Wx: Array, Wh: Array, b: Array,
                 retain_cache: bool) -> Tuple[Array, Optional[RNNCache]]:
    n, t, d = seq.shape
    h_dim = Wh.shape[0]
    h_states = np.zeros((t + 1, n, h_dim), dtype=np.float32)

    for step in range(t):
        pre = seq[:, step, :] @ Wx.T + h_states[step] @ Wh.T + b
        h_states[step + 1] = np.tanh(pre)

    cache = None
    if retain_cache:
        cache = RNNCache(x_seq=seq, h_states=h_states, Wx=Wx, Wh=Wh)
    return h_states[-1], cache


def _rnn_backward(dh_last: Array, cache: RNNCache) -> Tuple[Array, Array, Array, Array]:
    seq = cache.x_seq
    h_states = cache.h_states
    Wx = cache.Wx
    Wh = cache.Wh
    n, t, d = seq.shape
    h_dim = Wh.shape[0]

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros(h_dim, dtype=np.float32)
    dseq = np.zeros_like(seq)
    dh_next = dh_last

    for step in reversed(range(t)):
        h_curr = h_states[step + 1]
        h_prev = h_states[step]
        dz = dh_next * (1.0 - h_curr ** 2)
        dWx += dz.T @ seq[:, step, :]
        dWh += dz.T @ h_prev
        db += dz.sum(axis=0)
        dseq[:, step, :] = dz @ Wx
        dh_next = dz @ Wh

    return dseq, dWx, dWh, db


class CNNRNNClassifier:
    """Minimal CNN + RNN hybrid classifier implemented with NumPy only."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        conv_channels: Tuple[int, int] = (32, 64),
        hidden_size: int = 128,
        num_classes: int = 10,
        seed: int = 42,
    ) -> None:
        self.config = {
            "input_shape": list(input_shape),
            "conv_channels": list(conv_channels),
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "seed": seed,
        }
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()

    def _init_params(self) -> Dict[str, Array]:
        c_in, h, w = self.config["input_shape"]
        c1, c2 = self.config["conv_channels"]
        hidden = self.config["hidden_size"]
        num_classes = self.config["num_classes"]

        params = {
            "conv1_w": _kaiming((c1, c_in, 3, 3), c_in * 3 * 3, self.rng),
            "conv1_b": _zeros((c1,)),
            "conv2_w": _kaiming((c2, c1, 3, 3), c1 * 3 * 3, self.rng),
            "conv2_b": _zeros((c2,)),
        }

        h2 = h // 4
        w2 = w // 4
        seq_feat_dim = c2 * w2
        params.update({
            "rnn_wx": _kaiming((hidden, seq_feat_dim), seq_feat_dim, self.rng),
            "rnn_wh": _kaiming((hidden, hidden), hidden, self.rng),
            "rnn_b": _zeros((hidden,)),
            "fc_w": _kaiming((num_classes, hidden), hidden, self.rng),
            "fc_b": _zeros((num_classes,)),
        })
        return params

    def forward(self, x: Array, retain_cache: bool = True) -> Tuple[Array, Dict[str, object]]:
        x = x.astype(np.float32, copy=False)
        caches: Dict[str, object] = {}

        z1, cache1 = _conv_forward(x, self.params["conv1_w"], self.params["conv1_b"], 1, 1, retain_cache)
        a1, relu1 = _relu_forward(z1, retain_cache)
        p1, pool1 = _maxpool_forward(a1, pool_size=2, stride=2, retain_cache=retain_cache)

        z2, cache2 = _conv_forward(p1, self.params["conv2_w"], self.params["conv2_b"], 1, 1, retain_cache)
        a2, relu2 = _relu_forward(z2, retain_cache)
        p2, pool2 = _maxpool_forward(a2, pool_size=2, stride=2, retain_cache=retain_cache)

        n, c, h, w = p2.shape
        seq = p2.transpose(0, 2, 1, 3).reshape(n, h, c * w)
        last_h, rnn_cache = _rnn_forward(seq, self.params["rnn_wx"], self.params["rnn_wh"], self.params["rnn_b"], retain_cache)
        logits, fc_cache = _linear_forward(last_h, self.params["fc_w"], self.params["fc_b"], retain_cache)

        if retain_cache:
            caches = {
                "conv1": cache1,
                "relu1": relu1,
                "pool1": pool1,
                "conv2": cache2,
                "relu2": relu2,
                "pool2": pool2,
                "seq_shape": (n, c, h, w),
                "rnn": rnn_cache,
                "fc": fc_cache,
                "last_hidden": last_h,
            }
        else:
            caches = {"last_hidden": last_h, "seq_shape": (n, c, h, w)}
        return logits, caches

    def backward(self, grad_logits: Array, caches: Dict[str, object]) -> Dict[str, Array]:
        grads: Dict[str, Array] = {}

        grad_h, grads["fc_w"], grads["fc_b"] = _linear_backward(grad_logits, caches["fc"])
        seq_shape = caches["seq_shape"]
        grad_seq, grads["rnn_wx"], grads["rnn_wh"], grads["rnn_b"] = _rnn_backward(grad_h, caches["rnn"])
        n, c, h, w = seq_shape
        grad_p2 = grad_seq.reshape(n, h, c, w).transpose(0, 2, 1, 3)

        grad_a2 = _maxpool_backward(grad_p2, caches["pool2"])
        grad_z2 = _relu_backward(grad_a2, caches["relu2"])
        grad_p1, grads["conv2_w"], grads["conv2_b"] = _conv_backward(grad_z2, caches["conv2"])

        grad_a1 = _maxpool_backward(grad_p1, caches["pool1"])
        grad_z1 = _relu_backward(grad_a1, caches["relu1"])
        grad_x, grads["conv1_w"], grads["conv1_b"] = _conv_backward(grad_z1, caches["conv1"])
        _ = grad_x  # suppress unused warning
        return grads

    def apply_gradients(self, grads: Dict[str, Array], lr: float,
                        weight_decay: float = 0.0,
                        grad_clip: Optional[float] = None) -> float:
        total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if grad_clip is not None and total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-8)
            for key in grads:
                grads[key] = grads[key] * scale
            total_norm = grad_clip

        for name, param in self.params.items():
            grad = grads[name]
            if weight_decay > 0.0:
                grad = grad + weight_decay * param
            self.params[name] = param - lr * grad
        return total_norm

    def predict(self, x: Array) -> Array:
        logits, _ = self.forward(x, retain_cache=False)
        return np.argmax(logits, axis=1)

    def num_parameters(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.params.values())

    def state_dict(self) -> Dict[str, Array]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, Array]) -> None:
        for key in self.params:
            if key not in state:
                raise KeyError(f"Missing parameter: {key}")
            self.params[key] = state[key].astype(np.float32, copy=True)

    def save(self, path: str, epoch: int = 0, best_acc: float = 0.0,
             history: Optional[Dict[str, list]] = None) -> None:
        payload = {
            f"param_{k}": v for k, v in self.params.items()
        }
        payload["epoch"] = np.array([epoch], dtype=np.int32)
        payload["best_acc"] = np.array([best_acc], dtype=np.float32)
        payload["config"] = np.array([json.dumps(self.config)])
        if history is not None:
            payload["history"] = np.array([json.dumps(history)])
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> Tuple["CNNRNNClassifier", int, float, Optional[Dict[str, list]]]:
        data = np.load(path, allow_pickle=True)
        config = json.loads(str(data["config"][0]))
        model = cls(
            input_shape=tuple(config["input_shape"]),
            conv_channels=tuple(config["conv_channels"]),
            hidden_size=config["hidden_size"],
            num_classes=config["num_classes"],
            seed=config["seed"],
        )
        for key in model.params:
            model.params[key] = data[f"param_{key}"].astype(np.float32)
        epoch = int(data["epoch"][0]) if "epoch" in data else 0
        best_acc = float(data["best_acc"][0]) if "best_acc" in data else 0.0
        history = None
        if "history" in data:
            history = json.loads(str(data["history"][0]))
        return model, epoch, best_acc, history


def softmax_cross_entropy(logits: Array, targets: Array) -> Tuple[float, Array]:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), targets] + 1e-9).mean()
    grad = probs
    grad[np.arange(n), targets] -= 1.0
    grad /= n
    return float(loss), grad


def accuracy(logits: Array, targets: Array) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == targets) * 100.0)

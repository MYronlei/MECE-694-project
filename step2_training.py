# ==== Minimal, self-contained MLP Regressor (for CMAPSS RUL) ====
# Dependencies: numpy only
import numpy as np

# --- activations (derivatives expect ACTIVATED output) ---
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_prime(a): return a * (1.0 - a)          # a = sigmoid(z)

def tanh(x): return np.tanh(x)
def tanh_prime(a): return 1.0 - a**2                # a = tanh(z)

def relu(x): return np.maximum(0.0, x)
def relu_prime(a): return (a > 0.0).astype(float)   # a = relu(z)

_ACTS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh":    (tanh, tanh_prime),
    "relu":    (relu, relu_prime),
}

class MLPRegressorCustom:
    """
    Feed-forward MLP for regression with:
      - linear output layer
      - MSE or Huber loss
      - L1/L2 regularization (proper per-weight)
      - learning-rate decay
      - early stopping on validation loss (patience)
      - explicit biases per layer (no "append ones" tricks)
    """

    def __init__(
        self,
        size_hidden,                # list like [256,128,64]
        activation="relu",
        weight_init="kaiming",      # "xavier" | "kaiming" | "gaussian" | "uniform" | "zeros"
        learning_rate=1e-3,
        batch_size=256,
        l1=0.0,
        l2=1e-4,
        learning_rate_decay=0.0,    # LR_t = LR0 / (1 + t * decay)
        loss_fn="huber",            # "mse" | "huber"
        huber_delta=20.0,
        max_grad_norm=None,         # e.g., 5.0 for clipping
        random_state=None,
        verbose=False,
    ):
        self.size_hidden = list(size_hidden)
        self.activation_name = activation
        self.activation, self.activation_prime = _ACTS[activation]
        self.weight_init = weight_init
        self.lr0 = learning_rate
        self.batch_size = int(batch_size)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.decay = float(learning_rate_decay)
        self.loss_fn = loss_fn
        self.delta = float(huber_delta)
        self.max_grad_norm = max_grad_norm
        self.rng = np.random.default_rng(seed=random_state)
        self.verbose = verbose

        # learned params
        self.W = []   # weights (list of arrays)
        self.b = []   # biases  (list of arrays)

        # logs
        self.history_ = {"train_loss": [], "val_loss": []}

    # ---------- utils ----------
    def _init_layer(self, in_dim, out_dim):
        if self.weight_init == "zeros":
            W = np.zeros((in_dim, out_dim))
        elif self.weight_init == "gaussian":
            W = self.rng.normal(0.0, 1.0, size=(in_dim, out_dim))
        elif self.weight_init == "uniform":
            W = self.rng.uniform(-1.0, 1.0, size=(in_dim, out_dim))
        elif self.weight_init == "xavier":
            bound = np.sqrt(6.0 / (in_dim + out_dim))
            W = self.rng.uniform(-bound, bound, size=(in_dim, out_dim))
        elif self.weight_init == "kaiming":
            # good default for ReLU
            std = np.sqrt(2.0 / in_dim)
            W = self.rng.normal(0.0, std, size=(in_dim, out_dim))
        else:
            raise ValueError(f"Unknown weight_init: {self.weight_init}")
        b = np.zeros((1, out_dim))
        return W, b

    def _init_params(self, d_in):
        layer_sizes = [d_in] + self.size_hidden + [1]  # linear output
        self.W, self.b = [], []
        for i in range(len(layer_sizes) - 1):
            W, b = self._init_layer(layer_sizes[i], layer_sizes[i+1])
            self.W.append(W)
            self.b.append(b)

    def _forward(self, X):
        """
        Returns:
          acts: list of activations per layer (including input X and hidden activations; last item is output y_hat)
          zs:   list of pre-activations z for each layer (hidden + output)
        """
        a = X
        acts = [a]
        zs = []

        # hidden layers (nonlinear)
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = self.activation(z)
            zs.append(z)
            acts.append(a)

        # output layer: linear
        z = acts[-1] @ self.W[-1] + self.b[-1]
        y_hat = z  # linear
        zs.append(z)
        acts.append(y_hat)
        return acts, zs

    def _loss_and_grad_last(self, y_hat, y_true):
        # y_true, y_hat shapes: (B,1)
        if self.loss_fn == "mse":
            loss = 0.5 * np.mean((y_hat - y_true) ** 2)
            dL_dy = (y_hat - y_true) / y_true.shape[0]  # mean over batch
        elif self.loss_fn == "huber":
            e = y_hat - y_true
            abs_e = np.abs(e)
            quad = 0.5 * (e ** 2)
            lin  = self.delta * (abs_e - 0.5 * self.delta)
            loss = np.mean(np.where(abs_e <= self.delta, quad, lin))
            dL_dy = np.where(abs_e <= self.delta, e, self.delta * np.sign(e)) / y_true.shape[0]
        else:
            raise ValueError("loss_fn must be 'mse' or 'huber'")
        return loss, dL_dy

    def _regularize(self, grads_W):
        # add L1/L2 terms to weight grads (skip biases)
        if self.l2 > 0.0:
            grads_W = [gW + self.l2 * W for gW, W in zip(grads_W, self.W)]
        if self.l1 > 0.0:
            grads_W = [gW + self.l1 * np.sign(W) for gW, W in zip(grads_W, self.W)]
        return grads_W

    def _clip(self, gW, gb):
        if self.max_grad_norm is None:
            return gW, gb
        # global norm across all params
        total = 0.0
        for g in gW + gb:
            total += np.sum(g * g)
        total = np.sqrt(total)
        if total > self.max_grad_norm and total > 0.0:
            scale = self.max_grad_norm / total
            gW = [g * scale for g in gW]
            gb = [g * scale for g in gb]
        return gW, gb

    def _update(self, grads_W, grads_b, t):
        # decay schedule
        lr = self.lr0 / (1.0 + t * self.decay) if self.decay > 0.0 else self.lr0
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

    def _iterate_minibatches(self, X, y, batch_size, shuffle=True):
        n = X.shape[0]
        idx = np.arange(n)
        if shuffle:
            self.rng.shuffle(idx)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = idx[start:end]
            yield X[sel], y[sel]

    # ---------- public API ----------
    def fit(
        self, X_tr, y_tr, X_val=None, y_val=None,
        epochs=100, patience=10,
    ):
        """
        X_tr: (N, D)  y_tr: (N,) or (N,1)
        Early stopping uses validation loss if X_val provided; else trains full epochs.
        """
        y_tr = y_tr.reshape(-1, 1).astype(float)
        if X_val is not None:
            y_val = y_val.reshape(-1, 1).astype(float)

        self._init_params(X_tr.shape[1])
        best_loss = np.inf
        best_params = None
        no_improve = 0
        step = 0

        for epoch in range(1, epochs + 1):
            # ---- train one epoch
            train_losses = []
            for xb, yb in self._iterate_minibatches(X_tr, y_tr, self.batch_size, shuffle=True):
                # forward
                acts, zs = self._forward(xb)
                y_hat = acts[-1]

                # loss + dL/dy
                loss, dL_dy = self._loss_and_grad_last(y_hat, yb)
                train_losses.append(loss)

                # backward
                grads_W = [None] * len(self.W)
                grads_b = [None] * len(self.b)

                # output layer grads
                a_prev = acts[-2]                   # activation of last hidden (or X)
                grads_W[-1] = a_prev.T @ dL_dy      # (H,1)
                grads_b[-1] = np.sum(dL_dy, axis=0, keepdims=True)  # (1,1)

                # backprop through hidden layers
                delta = dL_dy @ self.W[-1].T        # (B,H_last)
                for i in range(len(self.W) - 2, -1, -1):
                    # derivative wrt activated output of layer i
                    d_act = self.activation_prime(acts[i+1])       # acts[i+1] is activation at layer i
                    delta *= d_act                                 # elementwise
                    grads_W[i] = acts[i].T @ delta                 # (in_i, out_i)
                    grads_b[i] = np.sum(delta, axis=0, keepdims=True)
                    if i > 0:
                        delta = delta @ self.W[i].T                # move to previous layer space

                # regularize, clip, update
                grads_W = self._regularize(grads_W)
                grads_W, grads_b = self._clip(grads_W, grads_b)
                self._update(grads_W, grads_b, t=step)
                step += 1

            # epoch logs
            tr_loss = float(np.mean(train_losses))
            self.history_["train_loss"].append(tr_loss)

            # validation
            if X_val is not None:
                val_loss = float(self.loss(self.predict_raw(X_val), y_val))
                self.history_["val_loss"].append(val_loss)
                if self.verbose:
                    print(f"Epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f}")

                # early stopping
                if val_loss + 1e-9 < best_loss:
                    best_loss = val_loss
                    best_params = ([W.copy() for W in self.W], [b.copy() for b in self.b])
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch} (best val {best_loss:.4f})")
                        break
            else:
                if self.verbose:
                    print(f"Epoch {epoch:3d} | train {tr_loss:.4f}")

        # restore best params if we used validation
        if best_params is not None:
            self.W, self.b = best_params

        return self

    def predict_raw(self, X):
        return self._forward(X)[0][-1]  # last activation (y_hat), shape (N,1)

    def predict(self, X):
        return self.predict_raw(X).ravel()

    # unified loss for external calls
    def loss(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1)
        return self._loss_and_grad_last(y_pred, y_true)[0]



# step2_training.py
import numpy as np
from step1_ml_pipeline import load_cmapss_data, data_processing

# ---------- minimal single MLP regressor (from earlier) ----------
# (paste the MLPRegressorCustom class here exactly as I gave you before)

# ---------- get data from step1 and make a leak-free split ----------
def get_nn_data(dataname="FD001", val_frac=0.15, shuffle_engines=True, random_state=42):
    # 1) load + scale using step1
    train_df, test_df = load_cmapss_data(dataname)
    train_std, test_std, scaler = data_processing(train_df, test_df)

    # 2) features/targets
    exclude = ["engine_id", "cycle", "RUL"]
    feature_cols = [c for c in train_std.columns if c not in exclude]
    X_train_full = train_std[feature_cols].to_numpy()
    y_train_full = train_std["RUL"].to_numpy()
    X_test = test_std[feature_cols].to_numpy()
    y_test = test_std["RUL"].to_numpy()

    # 3) split by engine_id to avoid leakage
    eng_ids = train_std["engine_id"].unique()
    if shuffle_engines:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(eng_ids)
    cut = int((1.0 - val_frac) * len(eng_ids))
    val_ids = set(eng_ids[cut:])
    mask_val = train_std["engine_id"].isin(val_ids).to_numpy()

    X_tr, y_tr = X_train_full[~mask_val], y_train_full[~mask_val]
    X_val, y_val = X_train_full[mask_val],  y_train_full[mask_val]

    meta = dict(feature_cols=feature_cols, scaler=scaler, val_engine_ids=sorted(list(val_ids)))
    return X_tr, y_tr, X_val, y_val, X_test, y_test, meta

def main():
    X_tr, y_tr, X_val, y_val, X_test, y_test, meta = get_nn_data(
        dataname="FD001", val_frac=0.15, shuffle_engines=True, random_state=42
    )
    print(f"[info] features={len(meta['feature_cols'])}")
    print(f"[shapes] X_tr={X_tr.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

    model = MLPRegressorCustom(
        size_hidden=[256, 128, 64],
        activation="relu",
        weight_init="kaiming",
        learning_rate=1e-3,
        batch_size=256,
        l1=0.0, l2=1e-4,
        learning_rate_decay=0.0,
        loss_fn="huber", huber_delta=20.0,
        max_grad_norm=5.0,
        random_state=42,
        verbose=True,
    )
    model.fit(X_tr, y_tr, X_val, y_val, epochs=100, patience=10)

    y_pred = model.predict(X_test)
    mae  = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred)**2)))
    ss_res = float(np.sum((y_test - y_pred)**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test))**2))
    r2 = 1.0 - ss_res/ss_tot
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

if __name__ == "__main__":
    main()

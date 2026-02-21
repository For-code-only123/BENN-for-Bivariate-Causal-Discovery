import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def rbf_kernel(X, Y=None, sigma=None):
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    Gx = np.sum(X**2, axis=1).reshape(-1,1)
    Gy = np.sum(Y**2, axis=1).reshape(1,-1)
    D = Gx + Gy - 2*np.dot(X, Y.T)
    if sigma is None:
        med = np.median(D[D > 0])
        sigma = np.sqrt(0.5*med) if (med > 0 and np.isfinite(med)) else 1.0
    K = np.exp(-D / (2*sigma**2 + 1e-12))
    return K

def delta_kernel(x1, x2=None):
    x1 = np.asarray(x1).ravel()
    x2 = x1 if x2 is None else np.asarray(x2).ravel()
    return (x1[:, None] == x2[None, :]).astype(float)

def hsic_test(K, L, n_perm=1000, seed=None):
    rng = np.random.RandomState(seed)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    base = np.trace(K @ H @ L @ H) / (n - 1)**2
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        Lp = L[idx][:, idx]
        val = np.trace(K @ H @ Lp @ H) / (n - 1)**2
        if val >= base:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return base, p


class BeltRegressor(nn.Module):
    def __init__(self, p_input_dim=1, r1=30, d=1, r2=18, m_output_dim=25,dropout=0.35):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(p_input_dim, r1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(r1, r1),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.belt = nn.Linear(r1, d)

        self.head = nn.Sequential(
            nn.Linear(d, r2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(r2, m_output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.belt(x)
        out = self.head(z)
        return out,z


def train_model(
    model, loss_fn,
    Xtr, Ytr, Xva, Yva,
    lr=1e-4, wd=5e-4, max_epochs=500, patience=30,
    batch_size=32, warmup_epochs=0,
    device='cuda'
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_state = None
    best_val = float('inf')
    wait = 0

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=device)
    Yva_t = torch.tensor(Yva, dtype=torch.float32, device=device)

    n_samples = Xtr_t.size(0)
    num_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(max_epochs):
        model.train()
        if epoch < warmup_epochs:
            current_lr = lr * (epoch + 1) / warmup_epochs
        else:
            current_lr = lr
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr

        perm = torch.randperm(n_samples, device=device)
        for i in range(num_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            xb = Xtr_t[idx]
            yb = Ytr_t[idx]

            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            outv, _ = model(Xva_t)
            vloss = loss_fn(outv, Yva_t).item()

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model



def get_hyperparams(n):
    if n <= 400:
        lr = 1e-3
        weight_decay = 1e-4
        max_epochs = 500
        patience = 25
        batch_size = 32
        warmup_epochs = 8
    elif n <= 800:
        lr = 5e-4
        weight_decay = 5e-5
        max_epochs = 600
        patience = 30
        batch_size = 48
        warmup_epochs = 5
    else:
        lr = 2e-4
        weight_decay = 2e-5
        max_epochs = 800
        patience = 35
        batch_size = 64
        warmup_epochs = 5

    min_lr = lr * 0.05

    return {
        'lr': lr,
        'weight_decay': weight_decay,
        'max_epochs': max_epochs,
        'patience': patience,
        'batch_size': batch_size,
        'warmup_epochs': warmup_epochs,
        'min_lr': min_lr
    }
def get_model_structure(n, num_classes):
    if n <= 300:
        r1 = 16
        r2 = max(16, num_classes * 2)
        dropout = 0.4
    elif n <= 600:
        r1 = 24
        r2 = max(32, num_classes * 3)
        dropout = 0.3
    else:
        r1 = 48
        r2 = max(64, num_classes * 4)
        dropout = 0.2

    return {
        'r1_layer': r1,
        'r2_layer': r2,
        'dropout': dropout
    }


def sorted_label_encode(arr):
    unique_vals = np.unique(arr)
    sorted_vals = np.sort(unique_vals)
    mapper = {val: idx for idx, val in enumerate(sorted_vals)}
    encoded = np.vectorize(mapper.get)(arr)
    return encoded, mapper


def direction_belt_1d(
    X, Y, X_type="disc", Y_type="disc",
    test_size=0.15, val_size=0.15, seed=None, device='cuda'
):
    rng = np.random.RandomState(seed)

    def norm(v):
        v = v.astype(float)
        return (v - v.mean()) / (v.std() + 1e-8)

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    def shift_to_non_negative(arr):
        """Shift array so that the smallest value becomes 0."""
        min_val = arr.min()
        if min_val < 0:
            arr = arr + abs(min_val)
        return arr

    # --- data prep ---

    X_shifted = shift_to_non_negative(X)
    Xc, X_mapper = sorted_label_encode(X_shifted)

    Y_shifted = shift_to_non_negative(Y)
    Yc, Y_mapper = sorted_label_encode(Y_shifted)

    Xfea = Xc.reshape(-1, 1)
    Yfea = Yc.reshape(-1, 1)

    n = len(Xfea)
    num_classes_X = len(X_mapper)
    num_classes_Y = len(Y_mapper)

    X_int = Xfea.astype(int).ravel()
    X_onehot = np.zeros((n, num_classes_X), dtype=np.float32)
    X_onehot[np.arange(n), X_int] = 1.0

    Y_int = Yfea.astype(int).ravel()
    Y_onehot = np.zeros((n, num_classes_Y), dtype=np.float32)
    Y_onehot[np.arange(n), Y_int] = 1.0

    # ===========================================================
    # A : X -> Y
    # ===========================================================
    Xa_tr, Xa_te, Ya_tr, Ya_te = train_test_split(
        X_onehot, Y_onehot, test_size=test_size, random_state=seed
    )
    Xa_tr, Xa_va, Ya_tr, Ya_va = train_test_split(
        Xa_tr, Ya_tr, test_size=val_size, random_state=seed
    )

    hp1 = get_model_structure(n,num_classes_Y)
    hp2 = get_hyperparams(n)

    model_A = BeltRegressor(
        p_input_dim=num_classes_X,
        r1=hp1['r1_layer'],
        d=1,
        r2=hp1['r2_layer'],
        m_output_dim=num_classes_Y,
        dropout=hp1['dropout']
    )

    loss_A = nn.MSELoss()
    model_A = train_model(
        model_A, loss_A,
        Xa_tr, Ya_tr, Xa_va, Ya_va,
        lr=hp2['lr'], wd=hp2['weight_decay'],
        max_epochs=hp2['max_epochs'], patience=hp2['patience'],
        batch_size =hp2['batch_size'],
        warmup_epochs=hp2['warmup_epochs'],
        device=device
    )

    eps = 1e-12
    T = 0.00001
    model_A.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xa_te, dtype=torch.float32, device=device)
        logits, _ = model_A(Xte_t)
        probs = F.softmax(logits/T, dim=1)
        probs_np = probs.cpu().numpy()

    true_idx = np.argmax(Ya_te, axis=1).astype(int)  # (n_te,)
    n_te, K = probs_np.shape
    probs_np = np.clip(probs_np, eps, 1.0)
    probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
    p_true = probs_np[np.arange(n_te), true_idx]  # (n_te,)
    p_true = np.clip(p_true, eps, 1.0)
    cdf_lower = np.cumsum(probs_np, axis=1) - probs_np  # (n_te, K)
    F_lower = cdf_lower[np.arange(n_te), true_idx]  # (n_te,)
    V = rng.uniform(0.0, 1.0, size=n_te)
    U = F_lower + V * p_true
    U = np.clip(U, eps, 1.0 - eps)
    rA1 = U.reshape(-1, 1)


    K_A = delta_kernel(np.argmax(Xa_te, axis=1))
    stat_A1, p_A1 = hsic_test(K_A, rbf_kernel(rA1))

    # ===========================================================
    # B : Y -> X
    # ===========================================================
    Xb_tr, Xb_te, Yb_tr, Yb_te = train_test_split(Y_onehot, X_onehot,  test_size=test_size, random_state=seed)
    Xb_tr, Xb_va, Yb_tr, Yb_va = train_test_split(Xb_tr, Yb_tr, test_size=val_size, random_state=seed)

    hp1 = get_model_structure(n,num_classes_X)
    hp2 = get_hyperparams(n)

    model_B = BeltRegressor(
        p_input_dim=num_classes_Y,
        r1=hp1['r1_layer'],
        d=1,
        r2=hp1['r2_layer'],
        m_output_dim=num_classes_X,
        dropout=hp1['dropout']
    )

    loss_B = nn.MSELoss()
    model_B = train_model(
        model_B, loss_B,
        Xb_tr, Yb_tr, Xb_va, Yb_va,
        lr=hp2['lr'], wd=hp2['weight_decay'],
        max_epochs=hp2['max_epochs'], patience=hp2['patience'],
        batch_size=hp2['batch_size'],
        warmup_epochs=hp2['warmup_epochs'],
        device=device
    )

    eps = 1e-12
    model_B.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xb_te, dtype=torch.float32, device=device)
        logits, _ = model_B(Xte_t)
        probs = F.softmax(logits/T, dim=1)
        probs_np = probs.cpu().numpy()

    true_idx = np.argmax(Yb_te, axis=1).astype(int)  # (n_te,)
    n_te, K = probs_np.shape
    probs_np = np.clip(probs_np, eps, 1.0)
    probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
    p_true = probs_np[np.arange(n_te), true_idx]  # (n_te,)
    p_true = np.clip(p_true, eps, 1.0)
    cdf_lower = np.cumsum(probs_np, axis=1) - probs_np  # (n_te, K)
    F_lower = cdf_lower[np.arange(n_te), true_idx]  # (n_te,)
    V = rng.uniform(0.0, 1.0, size=n_te)
    U = F_lower + V * p_true
    U = np.clip(U, eps, 1.0 - eps)
    rB1 = U.reshape(-1, 1)

    K_B = delta_kernel(np.argmax(Xb_te, axis=1))
    stat_B1, p_B1 = hsic_test(K_B, rbf_kernel(rB1))

    # ---------- decision ----------
    def decide(pA, pB, statA, statB):
        if pA > pB:
            return "X->Y"
        elif pB > pA:
            return "Y->X"
        else:
            return "X->Y" if statA < statB else "Y->X"

    decision = decide(p_A1, p_B1, stat_A1, stat_B1)

    return {
        "BENN-CD": {
            "p_A": p_A1, "stat_A": stat_A1,
            "p_B": p_B1, "stat_B": stat_B1,
            "decision": decision
        }
    }
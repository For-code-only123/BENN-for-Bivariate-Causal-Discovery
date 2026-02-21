import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
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

def hsic_test(K, L, n_perm=800, seed=None):
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

def gaussian_embedding_random_centers(Y, centers, sigma):
    Y = np.asarray(Y).reshape(-1, 1)                # (n,1)
    Phi = np.exp(- (Y - centers)**2 / (2.0 * sigma**2))
    return Phi

def train_model_CD(
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

def train_model_DC(
    model, loss_fn,
    Xtr, Ytr, Xva, Yva,
    centers, sigma,
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

    Ytr_emb = gaussian_embedding_random_centers(Ytr, centers, sigma)
    Yva_emb = gaussian_embedding_random_centers(Yva, centers, sigma)

    Ytr_t = torch.tensor(Ytr_emb, dtype=torch.float32, device=device)
    Yva_t = torch.tensor(Yva_emb, dtype=torch.float32, device=device)

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
    
def get_model_structure(n):
    if n <= 300:
        r1 = 16
        r2 = 40
        dropout = 0.45
        weight_decay = 1e-4
        m_output_dim = 20
    elif n <= 600:
        r1 = 24
        r2 = 80
        dropout = 0.35
        weight_decay = 5e-5
        m_output_dim=40
    else:
        r1 = 48
        r2 = 192
        dropout = 0.25
        weight_decay = 1e-5
        m_output_dim = 96

    return {
        'r1_layer': r1,
        'r2_layer': r2,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'm_output_dim': m_output_dim
    }

def get_model_structure_dis(n, num_classes):
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
    X, Y, X_type="cont", Y_type="discrete",
    test_size=0.15, val_size=0.15, seed=None, device='cuda'
):
    rng = np.random.RandomState(seed)

    def shift_to_non_negative(arr):
        """Shift array so that the smallest value becomes 0."""
        min_val = arr.min()
        if min_val < 0:
            arr = arr + abs(min_val)
        return arr
    # ---------- helpers ----------
    def norm(v):
        v = v.astype(float)
        return (v - v.mean())/(v.std() + 1e-8)

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    @torch.no_grad()
    def belt_embed(model, Xnp):
        X_t = to_t(Xnp)
        _, z = model(X_t)
        return z.cpu().numpy()

    class TinyRegressor(nn.Module):
        def __init__(self, input_dim=1, hidden1=16, hidden2=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, 1),  
            )

        def forward(self, x):
            return self.net(x)

    def train_tiny_regressor(z_tr, y_tr, z_va, y_va, n, device='cuda'):
        if n <= 500:
            hidden1, hidden2 = 16, 16
            lr = 5e-4
            weight_decay = 5e-5
            patience = 25
            max_epochs = 600
        else:
            hidden1, hidden2 = 32, 32
            lr = 2e-4
            weight_decay = 5e-5
            patience = 30
            max_epochs = 600

        model = TinyRegressor(input_dim=z_tr.shape[1], hidden1=hidden1, hidden2=hidden2).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        Ztr, Ytr = to_t(z_tr), to_t(y_tr.reshape(-1, 1))
        Zva, Yva = to_t(z_va), to_t(y_va.reshape(-1, 1))
        best, best_loss, wait = None, float('inf'), 0
        for _ in range(max_epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(Ztr), Ytr)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(Zva), Yva).item()
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                wait = 0
                best = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best is not None:
            model.load_state_dict(best)
        model.eval()
        return model

    # ---------- data prep ----------
    Xc = norm(X) if X_type == "cont" else X.astype(int)
    Y_shifted = shift_to_non_negative(Y)
    Yc, Y_mapper = sorted_label_encode(Y_shifted)

    n = len(Xc)

    hp1 = get_model_structure(n)

    Xfea = Xc.reshape(-1, 1)
    Yfea = Yc.reshape(-1, 1)
    mu_X = float(np.mean(Xc))
    std_X = float(np.std(Xc))
    sigma_X = max(std_X, 1e-8)  

    lo_X, hi_X = mu_X - 2.0 * sigma_X, mu_X + 2.0 * sigma_X
    m = hp1['m_output_dim']
    X_centers = rng.uniform(lo_X, hi_X, size=(1, m))

    num_classes_Y = len(Y_mapper)  
    Y_int = Yfea.astype(int).ravel()
    Y_onehot = np.zeros((n, num_classes_Y), dtype=np.float32)
    Y_onehot[np.arange(n), Y_int] = 1.0
    # ===========================================================
    # A : X -> Y
    # ===========================================================
    Xa_tr, Xa_te, Ya_tr, Ya_te = train_test_split(Xfea, Y_onehot, test_size=test_size, random_state=seed)
    Xa_tr, Xa_va, Ya_tr, Ya_va = train_test_split(Xa_tr, Ya_tr, test_size=val_size, random_state=seed)

    hp1 = get_model_structure_dis(n,num_classes_Y)
    hp2 = get_hyperparams(n)

    model_A = BeltRegressor(
        p_input_dim=1,
        r1=hp1['r1_layer'],
        d=1,
        r2=hp1['r2_layer'],
        m_output_dim=num_classes_Y,
        dropout=hp1['dropout']
    )

    loss_A = nn.MSELoss()
    model_A = train_model_CD(
        model_A, loss_A,
        Xa_tr, Ya_tr, Xa_va, Ya_va,
        lr=hp2['lr'], wd=hp2['weight_decay'],
        max_epochs=hp2['max_epochs'], patience=hp2['patience'],
        batch_size =hp2['batch_size'],
        warmup_epochs=hp2['warmup_epochs'],
        device=device
    )
    
    eps = 1e-12
    model_A.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xa_te, dtype=torch.float32, device=device)
        logits, _ = model_A(Xte_t)  
        probs = F.softmax(logits, dim=1)  
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

    K_A = rbf_kernel(Xa_te.reshape(-1, 1))
    stat_A1, p_A1 = hsic_test(K_A, rbf_kernel(rA1))

    # ===========================================================
    # B : Y -> X
    # ===========================================================
    Xb_tr, Xb_te, Yb_tr, Yb_te = train_test_split(Y_onehot, Xc, test_size=test_size, random_state=seed)
    Xb_tr, Xb_va, Yb_tr, Yb_va = train_test_split(Xb_tr, Yb_tr, test_size=val_size, random_state=seed)

    hp1 = get_model_structure(n)
    hp2 = get_hyperparams(n)

    model_B = BeltRegressor(
        p_input_dim=num_classes_Y,
        r1=hp1['r1_layer'],
        d=1,
        r2=hp1['r2_layer'],
        m_output_dim=hp1['m_output_dim'],
        dropout=hp1['dropout']
    )

    loss_B = nn.MSELoss()
    model_B = train_model_DC(
        model_B, loss_B,
        Xb_tr, Yb_tr, Xb_va, Yb_va,
        lr=hp2['lr'], wd=hp2['weight_decay'],
        max_epochs=hp2['max_epochs'], patience=hp2['patience'],
        batch_size=hp2['batch_size'],
        warmup_epochs=hp2['warmup_epochs'],centers =  X_centers,sigma = sigma_X,
        device=device
    )

    zB_tr = belt_embed(model_B, Xb_tr)
    zB_va = belt_embed(model_B, Xb_va)
    zB_te = belt_embed(model_B, Xb_te)

    nnB = train_tiny_regressor(zB_tr, Yb_tr, zB_va, Yb_va, n)
    with torch.no_grad():
        rB1 = (to_t(Yb_te.reshape(-1, 1)) - nnB(to_t(zB_te))).cpu().numpy()

    zB_rf = np.vstack([zB_tr, zB_va])
    yB_rf = np.concatenate([Yb_tr, Yb_va], axis=0).ravel()
    rfB = RandomForestRegressor(n_estimators=200, min_samples_leaf=6, n_jobs=-1, random_state=seed)
    rfB.fit(zB_rf, yB_rf)
    rB2 = Yb_te.reshape(-1, 1) - rfB.predict(zB_te).reshape(-1, 1)

    K_B = delta_kernel(np.argmax(Xb_te, axis=1))
    stat_B1, p_B1 = hsic_test(K_B, rbf_kernel(rB1))
    stat_B2, p_B2 = hsic_test(K_B, rbf_kernel(rB2))

    # ---------- decision ----------
    def decide(pA, pB, statA, statB):
        if pA > pB:
            return "X->Y"
        elif pB > pA:
            return "Y->X"
        else:
            return "X->Y" if statA < statB else "Y->X"

    decision_nn = decide(p_A1, p_B1, stat_A1, stat_B1)
    decision_rf = decide(p_A1, p_B2, stat_A1, stat_B2)

    return {
        "BENN-NN": {
            "p_A": p_A1, "stat_A": stat_A1,
            "p_B": p_B1, "stat_B": stat_B1,
            "decision": decision_nn
        },
        "BENN-RF": {
            "p_A": p_A1, "stat_A": stat_A1,
            "p_B": p_B2, "stat_B": stat_B2,
            "decision": decision_rf
        }
    }



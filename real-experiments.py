import os
import glob
import numpy as np
import pandas as pd
import importlib.util
import warnings
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gamma
warnings.filterwarnings("ignore")

def delta_kernel(X):
    X = X.reshape(-1, 1)
    return (X == X.T).astype(float)

def get_kernel(data, var_type):
    data = data.reshape(-1, 1)
    if var_type == 'disc':
        return delta_kernel(data)
    else:
        return rbf_kernel(data)

def hsic_test(X, Y, X_type, Y_type, alpha=0.05, n_perms=1000):

    n = X.shape[0]

    if n < 5: return False

    K = get_kernel(X, X_type)
    L = get_kernel(Y, Y_type)

    def center_kernel(M):
        row_mean = M.mean(axis=0, keepdims=True)
        col_mean = M.mean(axis=1, keepdims=True)
        grand_mean = M.mean()
        return M - row_mean - col_mean + grand_mean

    Kc = center_kernel(K)
    Lc = center_kernel(L)

    test_stat = np.sum(Kc * Lc)

    if n > 1000:

        mu_null = np.trace(Kc) * np.trace(Lc) / n
        var_null = 2 * np.sum(Kc ** 2) * np.sum(Lc ** 2) / (n ** 2)

        if var_null == 0: return False

        k = mu_null ** 2 / var_null  # shape
        theta = var_null / mu_null  # scale

        p_value = gamma.sf(test_stat, a=k, scale=theta)

    else:
        hsic_perms = np.zeros(n_perms)
        idx = np.arange(n)
        for i in range(n_perms):
            np.random.shuffle(idx)
            Lc_perm = Lc[np.ix_(idx, idx)]
            hsic_perms[i] = np.sum(Kc * Lc_perm)

        p_value = (np.sum(hsic_perms >= test_stat) + 1) / (n_perms + 1)

    return p_value < alpha

def load_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[Error] Failed to load {module_name}: {e}")
        return None

# Load Modules
# Continuous
PATH_CONT = r"\path\BENN_Continuous.py"
mod_benn_cont = load_module_from_path("benn_cont", PATH_CONT)

# Discrete 1 (BENN_DIS)
PATH_DISC_1 = r"\path\BENN_DIS.py"
mod_benn_dis1 = load_module_from_path("benn_dis1", PATH_DISC_1)

# Discrete 2 (BENN_DS)
PATH_DISC_2 = r"\path\BENN_DS.py"
mod_benn_dis2 = load_module_from_path("benn_dis2", PATH_DISC_2)

# Mixed
PATH_MIXED = r"\path\BENN_Mixed.py"
mod_benn_mixed = load_module_from_path("benn_mixed", PATH_MIXED)

def get_target_file(data_folder):
    priorities = ["*mixed.numeric*.txt", "*mixed.maximum*.txt", "*mixed.max*.txt"]
    for pattern in priorities:
        files = glob.glob(os.path.join(data_folder, pattern))
        if files: return files[0]
    return None

def determine_type(series):
    if series.dtype == 'object' or series.dtype.name == 'category':
        return 'disc'
    if series.nunique() <= 10:
        return 'disc'
    return 'cont'

def process_datasets():
    base_dir = r"\path\real-datasets"
    output_base = os.path.join(base_dir, "BENN")

    # Create Output Folders
    for t in ["con", "dis", "mixed"]:
        os.makedirs(os.path.join(output_base, t), exist_ok=True)

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() and f.name not in ["benn", "lim"]]
    print(f"[Init] Found {len(subfolders)} datasets.")

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        data_path = os.path.join(folder_path, "data")
        target_file = get_target_file(data_path)

        if not target_file:
            continue

        print(f"\nProcessing Dataset: {folder_name}")

        try:
            # 1. Load Data
            try:
                df = pd.read_csv(target_file, sep=None, engine='python')
            except:
                df = pd.read_csv(target_file, delim_whitespace=True)

            df.replace('?', np.nan, inplace=True)
            df.dropna(inplace=True)

            # Rule for Algerian Forest
            if 'algerian-forest-fires' in folder_name.lower():
                cols_to_drop = [c for c in df.columns if c.lower() in ['day', 'year']]
                df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # 2. Encode & Type Identification
            df_encoded = df.copy()
            col_types = {}
            for col in df.columns:
                ctype = determine_type(df[col])
                col_types[col] = ctype
                if ctype == 'disc':
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df[col].astype(str))
                else:
                    df_encoded[col] = df[col].astype(float)

            # 3. Pairwise Analysis - Storage Initialization
            results_store = {
                "con": {"BENN-NN": [], "BENN-RF": []},
                "dis": {"BENN_DIS": [], "BENN_DS": []},
                "mixed": {"BENN-NN": [], "BENN-RF": []}
            }

            pairs = list(combinations(df.columns, 2))

            for col_a, col_b in pairs:
                type_a = col_types[col_a]
                type_b = col_types[col_b]

                data_a = df_encoded[col_a].values
                data_b = df_encoded[col_b].values

                if not hsic_test(data_a, data_b, type_a, type_b):
                    continue

                if type_a == 'cont' and type_b == 'cont':
                    if mod_benn_cont:
                        res = mod_benn_cont.direction_belt_1d(
                            data_a, data_b, X_type="cont", Y_type="cont",
                            test_size=0.15, val_size=0.15, seed=42, device='cuda'
                        )

                        for key in ["BENN-NN", "BENN-RF"]:
                            if key in res:
                                d = res[key]
                                cause = col_a if d['decision'] == "X->Y" else col_b
                                effect = col_b if d['decision'] == "X->Y" else col_a
                                results_store["con"][key].append({
                                    "Cause": cause, "Effect": effect,
                                    "p_A": d['p_A'], "stat_A": d['stat_A'],
                                    "p_B": d['p_B'], "stat_B": d['stat_B']
                                })

                elif type_a == 'disc' and type_b == 'disc':
                    if mod_benn_dis1:
                        res1 = mod_benn_dis1.direction_belt_1d(
                            data_a, data_b, X_type="disc", Y_type="disc",
                            test_size=0.15, val_size=0.15, seed=42, device='cuda'
                        )
                        if "BENN-CD" in res1:
                            d = res1["BENN-CD"]
                            cause = col_a if d['decision'] == "X->Y" else col_b
                            effect = col_b if d['decision'] == "X->Y" else col_a
                            results_store["dis"]["BENN_DIS"].append({
                                "Cause": cause, "Effect": effect,
                                "p_A": d['p_A'], "stat_A": d['stat_A'],
                                "p_B": d['p_B'], "stat_B": d['stat_B']
                            })

                    if mod_benn_dis2:
                        res2 = mod_benn_dis2.direction_belt_1d(
                            data_a, data_b, X_type="disc", Y_type="disc",
                            test_size=0.15, val_size=0.15, seed=42, device='cuda'
                        )
                        if res2:
                            k = list(res2.keys())[0]
                            d = res2[k]
                            cause = col_a if d['decision'] == "X->Y" else col_b
                            effect = col_b if d['decision'] == "X->Y" else col_a
                            results_store["dis"]["BENN_DS"].append({
                                "Cause": cause, "Effect": effect,
                                "p_A": d['p_A'], "stat_A": d['stat_A'],
                                "p_B": d['p_B'], "stat_B": d['stat_B']
                            })

                else:
                    if type_a == 'cont' and type_b == 'disc':
                        X_in, Y_in = data_a, data_b
                        name_X, name_Y = col_a, col_b
                    else:
                        X_in, Y_in = data_b, data_a
                        name_X, name_Y = col_b, col_a

                    if mod_benn_mixed:
                        res = mod_benn_mixed.direction_belt_1d(
                            X_in, Y_in, X_type="cont", Y_type="discrete",
                            test_size=0.15, val_size=0.15, seed=42, device='cuda'
                        )
                        for key in ["BENN-NN", "BENN-RF"]:
                            if key in res:
                                d = res[key]
                                cause = name_X if d['decision'] == "X->Y" else name_Y
                                effect = name_Y if d['decision'] == "X->Y" else name_X

                                results_store["mixed"][key].append({
                                    "Cause": cause, "Effect": effect,
                                    "p_A": d['p_A'], "stat_A": d['stat_A'],
                                    "p_B": d['p_B'], "stat_B": d['stat_B']
                                })

            for m_name, rows in results_store["con"].items():
                if rows:
                    safe_name = m_name.replace('-', '_')
                    fname = f"{safe_name}_{folder_name}.csv"
                    pd.DataFrame(rows).to_csv(os.path.join(output_base, "con", fname), index=False)

            for m_name, rows in results_store["dis"].items():
                if rows:
                    safe_name = m_name.replace('-', '_')
                    fname = f"{safe_name}_{folder_name}.csv"
                    pd.DataFrame(rows).to_csv(os.path.join(output_base, "dis", fname), index=False)

            for m_name, rows in results_store["mixed"].items():
                if rows:
                    safe_name = m_name.replace('-', '_')
                    fname = f"{safe_name}_{folder_name}.csv"
                    pd.DataFrame(rows).to_csv(os.path.join(output_base, "mixed", fname), index=False)

            print(f"  -> Finished {folder_name}")

        except Exception as e:
            print(f"  -> [ERROR] Processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    process_datasets()

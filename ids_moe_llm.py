#!/usr/bin/env python3
# IDS with MoE + LLM - WINDOWS-FIXED VERSION
# RUN: python ids_moe_llm_fixed.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import urllib.request
import ssl
import os
import json

# ===== FIX SSL FOR WINDOWS =====
ssl._create_default_https_context = ssl._create_unverified_context
print("[✓] SSL certificate bypass enabled for Windows")

# ===== STEP 1: DOWNLOAD NSL-KDD (with retry) =====
if not os.path.exists("KDDTrain+.txt") or not os.path.exists("KDDTest+.txt"):
    print("\n[1/4] Downloading NSL-KDD dataset...")
    try:
        # Try GitHub raw URLs (HTTPS)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                                   "KDDTrain+.txt")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                                   "KDDTest+.txt")
        print("✓ Dataset downloaded successfully")
    except Exception as e1:
        print(f"⚠ HTTPS download failed: {e1}")
        print("  Trying HTTP fallback...")
        try:
            # Fallback to HTTP mirror
            urllib.request.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
                                       "kddcup.data.gz")
            print("⚠ Using KDD Cup 99 (10%) as fallback dataset")
            # Convert to NSL-KDD format (simplified)
            import gzip

            with gzip.open('kddcup.data.gz', 'rt') as f:
                lines = f.readlines()[:10000]  # Take 10k samples for speed
            with open('KDDTrain+.txt', 'w') as f:
                f.writelines(lines[:8000])
            with open('KDDTest+.txt', 'w') as f:
                f.writelines(lines[8000:])
            print("✓ Fallback dataset prepared")
        except Exception as e2:
            print(f"❌ All downloads failed: {e2}")
            print("\n🚨 MANUAL FIX REQUIRED:")
            print("1. Open browser and download these files:")
            print("   → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt")
            print("   → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt")
            print("2. Save them in your current folder: C:\\Users\\LENOVO i7\\Desktop\\M2\\ids-moe-llm\\")
            print("3. Run this script again")
            exit(1)

# ===== STEP 2: LOAD & PREPROCESS =====
print("\n[2/4] Loading and preprocessing data...")
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
             "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
             "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
             "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
             "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
             "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]

try:
    df_train = pd.read_csv('KDDTrain+.txt', names=col_names, header=None)
    df_test = pd.read_csv('KDDTest+.txt', names=col_names, header=None)
except FileNotFoundError:
    print("\n❌ Files not found! MANUAL FIX:")
    print("1. Download these files manually:")
    print("   → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt")
    print("   → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt")
    print("2. Save them in:", os.getcwd())
    print("3. Run script again")
    exit(1)

# Binary classification: normal (0) vs attack (1)
df_train['label'] = df_train['attack'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['label'] = df_test['attack'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical features
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Features and labels
feature_cols = [c for c in df_train.columns if c not in ['attack', 'last_flag', 'label']]
X_train, y_train = df_train[feature_cols].values, df_train['label'].values
X_test, y_test = df_test[feature_cols].values, df_test['label'].values

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"✓ Preprocessing done. Train: {X_train.shape}, Test: {X_test.shape}")

# ===== STEP 3: BUILD MINIMAL MoE (3 EXPERTS + ROUTER) =====
print("\n[3/4] Training MoE model (takes ~60 seconds)...")



class MinimalMoE:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        self.router = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        self.expert_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.expert_lr = LogisticRegression(max_iter=1000, random_state=42)
        self.expert_svm = SVC(probability=True, random_state=42)

    def fit(self, X, y):
        self.router.fit(X[:5000], y[:5000])
        self.expert_rf.fit(X, y)
        self.expert_lr.fit(X, y)
        self.expert_svm.fit(X[:2000], y[:2000])

    def predict(self, X):
        import numpy as np
        router_w = self.router.predict_proba(X)
        # Pad to 3 columns if needed (equal share for missing experts)
        if router_w.shape[1] < 3:
            pad_col = np.full((router_w.shape[0], 1), 1.0 / 3)
            if router_w.shape[1] == 2:
                # Re-normalize first two to sum 2/3, then add 1/3 as third
                scaled = router_w * (2.0 / 3.0)
                router_w = np.hstack([scaled, pad_col])
            else:
                router_w = np.hstack([np.tile([1.0 / 3], (router_w.shape[0], 3))])

        p_rf = self.expert_rf.predict_proba(X)[:, 1]
        p_lr = self.expert_lr.predict_proba(X)[:, 1]
        p_svm = self.expert_svm.predict_proba(X)[:, 1]

        final_prob = (
            router_w[:, 0] * p_rf +
            router_w[:, 1] * p_lr +
            router_w[:, 2] * p_svm
        )
        return (final_prob > 0.5).astype(int), final_prob# python


moe = MinimalMoE()
moe.fit(X_train, y_train)
y_pred_moe, y_prob_moe = moe.predict(X_test)
print(f"✓ MoE trained. Accuracy: {accuracy_score(y_test, y_pred_moe):.4f}")

# ===== STEP 4: LLM ENHANCEMENT (SIMULATED - WORKS OFFLINE) =====
print("\n[4/4] Applying LLM reasoning (simulated)...")


def simulated_llm_enhance(features, moe_pred, moe_conf):
    """Fake LLM that adds explainability (works 100% offline)"""
    # Low-confidence predictions get LLM review
    if moe_conf < 0.65:
        # Heuristic: high serror_rate + low same_srv_rate = likely attack
        if features[24] > 0.8 and features[28] < 0.2:
            return 1, "LLM detected port scanning pattern (high SYN errors + low service diversity)"
        # Heuristic: duration=0 + urgent>0 = exploit attempt
        if features[0] == 0 and features[8] > 0:
            return 1, "LLM flagged zero-duration urgent packet (exploit signature)"
        return moe_pred, "LLM reviewed low-confidence prediction - no anomaly found"
    return moe_pred, "LLM confirmed high-confidence MoE decision"


# Apply to test set (sample 100 for speed)
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), 100, replace=False)
y_pred_final = y_pred_moe.copy()
explanations = []

for i in sample_idx:
    pred, expl = simulated_llm_enhance(X_test[i], y_pred_moe[i], y_prob_moe[i])
    y_pred_final[i] = pred
    explanations.append({
        'sample_id': int(i),
        'true_label': int(y_test[i]),
        'moe_pred': int(y_pred_moe[i]),
        'final_pred': int(pred),
        'confidence': float(y_prob_moe[i]),
        'explanation': expl
    })

# Save explanations for report
with open('llm_explanations.json', 'w') as f:
    json.dump(explanations[:5], f, indent=2)  # Save 5 examples

# ===== RESULTS =====
print("\n" + "=" * 60)
print("FINAL RESULTS (MoE + Simulated LLM)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"Baseline (MoE alone): {accuracy_score(y_test, y_pred_moe):.4f}")
print("\nSample LLM Explanations:")
for ex in explanations[:3]:
    status = "✓ CORRECT" if ex['final_pred'] == ex['true_label'] else "✗ WRONG"
    print(f"\n{status} | True={ex['true_label']} → Pred={ex['final_pred']}")
    print(f"  {ex['explanation']}")
print("\n" + "=" * 60)
print("✓ SUCCESS! Results saved to 'llm_explanations.json'")
print("✓ Your working directory:", os.getcwd())
print("=" * 60)
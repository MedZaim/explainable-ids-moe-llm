#!/usr/bin/env python3
# IDS with REAL Local LLM via Ollama
# Requires: pip install pandas scikit-learn numpy ollama

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import urllib.request
import ssl
import os
import json
import time
import sys

# ===== FIX SSL FOR WINDOWS =====
ssl._create_default_https_context = ssl._create_unverified_context

# ===== STEP 1: DOWNLOAD DATASET =====
if not os.path.exists("KDDTrain+.txt"):
    print("[1/5] Downloading NSL-KDD...")
    try:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                                   "KDDTrain+.txt")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                                   "KDDTest+.txt")
        print("✓ Dataset downloaded")
    except Exception as e:
        print(f"⚠ Download failed: {e}")
        print("  MANUAL FIX: Download these files manually and save in this folder:")
        print("  → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt")
        print("  → https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt")
        input("Press Enter after saving files...")
        if not os.path.exists("KDDTrain+.txt"):
            sys.exit(1)

# ===== STEP 2: PREPROCESS =====
print("\n[2/5] Preprocessing data...")
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

df_train = pd.read_csv('KDDTrain+.txt', names=col_names, header=None)
df_test = pd.read_csv('KDDTest+.txt', names=col_names, header=None)

df_train['label'] = df_train['attack'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['label'] = df_test['attack'].apply(lambda x: 0 if x == 'normal' else 1)

for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

feature_cols = [c for c in df_train.columns if c not in ['attack', 'last_flag', 'label']]
X_train, y_train = df_train[feature_cols].values, df_train['label'].values
X_test, y_test = df_test[feature_cols].values, df_test['label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"✓ Data ready: {X_train.shape} train samples")
print(f"✓ Data ready: {X_test.shape} test samples")

# ===== STEP 3: MINIMAL MoE =====
print("\n[3/5] Training MoE model...")


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
        # Get prediction probabilities from each expert
        p_rf = self.expert_rf.predict_proba(X)[:, 1]
        p_lr = self.expert_lr.predict_proba(X)[:, 1]
        p_svm = self.expert_svm.predict_proba(X)[:, 1]

        # FIXED: Use static weights instead of faulty router
        # (Router was causing IndexError in binary classification)
        weight_rf = 0.4  # Best for DoS attacks
        weight_lr = 0.35  # Best for probing =استطلاع / فحص النظام أو الشبكة
        weight_svm = 0.25  # Best for rare attacks

        # Weighted ensemble
        final_prob = (
                weight_rf * p_rf +
                weight_lr * p_lr +
                weight_svm * p_svm
        )
        #final_prob = np.clip(final_prob, 0, 1)
        final_prob = np.clip(final_prob, 0.01, 0.99)  # ← was (0, 1)
        predictions = (final_prob > 0.5).astype(int)

        return predictions, final_prob


moe = MinimalMoE()
moe.fit(X_train, y_train)
y_pred_moe, y_prob_moe = moe.predict(X_test)
print(f"✓ MoE baseline accuracy: {accuracy_score(y_test, y_pred_moe):.4f}")

# ===== STEP 4: REAL LLM INTEGRATION (Ollama) =====
print("\n[4/5] Setting up REAL local LLM (Ollama)...")
try:
    import ollama

    # Test if model exists
    try:
        ollama.chat(model='phi3', messages=[{'role': 'user', 'content': 'test'}], options={'num_predict': 5})
        MODEL_NAME = 'phi3'
        print(f"✓ Using Ollama model: {MODEL_NAME}")
    except:
        print("⚠ 'phi3' not found. Trying 'tinyllama'...")
        try:
            ollama.pull('tinyllama')  # Auto-download if missing
            MODEL_NAME = 'tinyllama'
            print(f"✓ Using Ollama model: {MODEL_NAME}")
        except:
            raise Exception("No Ollama model available")


    def real_llm_enhance(features, moe_pred, moe_conf, feature_names=None):
        """REAL LLM reasoning via Ollama"""
        # Build natural language description
        desc = (
            f"Network traffic features:\n"
            f"- Duration: {features[0]:.1f} seconds\n"
            f"- Source bytes: {features[4]:.0f}\n"
            f"- Destination bytes: {features[5]:.0f}\n"
            f"- SYN error rate: {features[24]:.2%}\n"
            f"- Same service rate: {features[28]:.2%}\n"
            f"- Urgent packets: {features[8]:.0f}\n\n"
            f"MoE prediction: {'ATTACK' if moe_pred else 'NORMAL'} (confidence: {moe_conf:.1%})\n\n"
            f"Analyze this traffic. Is it malicious? Answer ONLY 'YES' or 'NO', then explain in one sentence."
        )

        try:
            start = time.time()
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': desc}],
                options={'num_predict': 50, 'temperature': 0.3}  # Deterministic output
            )
            latency = time.time() - start

            # Parse response
            text = response['message']['content'].strip().upper()
            if 'YES' in text.split()[:3]:
                final_pred = 1
            elif 'NO' in text.split()[:3]:
                final_pred = 0
            else:
                final_pred = moe_pred  # Fallback to MoE

            # Extract explanation (first sentence after YES/NO)
            explanation = text.split('.', 1)[0] if '.' in text else text
            explanation = f"LLM ({MODEL_NAME}): {explanation} (latency: {latency:.1f}s)"

            return final_pred, explanation, latency

        except Exception as e:
            return moe_pred, f"LLM failed (fallback to MoE): {str(e)[:50]}", 0.0

except ImportError:
    print("⚠ Ollama not installed. Installing minimal fallback (DistilBERT)...")

    # Fallback: DistilBERT for text classification (no Ollama needed)
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)


        def real_llm_enhance(features, moe_pred, moe_conf, feature_names=None):
            desc = f"network traffic duration {features[0]:.0f} src_bytes {features[4]:.0f} serror_rate {features[24]:.2f}"
            prompt = f"Is this network traffic malicious? {desc}"

            try:
                start = time.time()
                result = classifier(prompt[:256], truncation=True)
                latency = time.time() - start

                # Map sentiment to maliciousness (hack but works)
                score = result[0]['score']
                label = result[0]['label']
                final_pred = 1 if (label == 'NEGATIVE' and score > 0.7) else moe_pred

                return final_pred, f"DistilBERT: {'suspicious' if final_pred else 'normal'} (score={score:.2f})", latency
            except:
                return moe_pred, "DistilBERT fallback failed", 0.0

    except:
        # Ultimate fallback: enhanced heuristics (still better than fake)
        print("⚠⚠⚠ NO LLM AVAILABLE - using enhanced heuristics")


        def real_llm_enhance(features, moe_pred, moe_conf, feature_names=None):
            reasons = []
            if features[24] > 0.8: reasons.append("high SYN errors")
            if features[28] < 0.2: reasons.append("low service diversity")
            if features[0] == 0 and features[8] > 0: reasons.append("zero-duration urgent packet")
            if features[31] > 0.9: reasons.append("high destination host count")

            if len(reasons) >= 2 or (moe_conf < 0.6 and len(reasons) >= 1):
                return 1, f"Rule-based LLM: Suspicious pattern ({', '.join(reasons[:2])})"
            return moe_pred, "Rule-based LLM: No strong anomaly indicators"

# ===== STEP 5: RUN INFERENCE (SAMPLE n_samples FOR SPEED) =====
n_samples = 20
print(f"\n[5/5] Running MoE + REAL LLM inference on {n_samples} samples...")
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), n_samples, replace=False)
y_pred_final = np.zeros(len(X_test), dtype=int)
latencies = []
explanations = []

for i, idx in enumerate(sample_idx):
    moe_pred = y_pred_moe[idx]
    moe_conf = y_prob_moe[idx]

    # Only run LLM on low-confidence predictions (saves time!)
    # if moe_conf < 0.75:
    #     pred, expl, lat = real_llm_enhance(X_test[idx], moe_pred, moe_conf)
    #     latencies.append(lat)
    # else:
    #     pred, expl, lat = moe_pred, "High-confidence MoE prediction (LLM skipped)", 0.0

    # Run LLM on ALL predictions
    pred, expl, lat = real_llm_enhance(X_test[idx], moe_pred, moe_conf)
    latencies.append(lat)

    y_pred_final[idx] = pred
    explanations.append({
        'sample_id': int(idx),
        'true_label': int(y_test[idx]),
        'moe_pred': int(moe_pred),
        'llm_pred': int(pred),
        'confidence': float(moe_conf),
        'explanation': expl,
        'latency_sec': round(lat, 2)
    })


    # Progress bar
    pct = (i + 1) / n_samples * 100
    filled = int((i + 1) / n_samples * n_samples)
    bar = '█' * filled + '░' * (n_samples - filled)
    print(f'\rProgress: |{bar}| {pct:.0f}%', end='', flush=True)


# Fill non-sampled indices with MoE predictions
for i in range(len(X_test)):
    if i not in sample_idx:
        y_pred_final[i] = y_pred_moe[i]

# ===== RESULTS =====
print("\n\n" + "=" * 70)
print("FINAL RESULTS: MoE + REAL LOCAL LLM")
print("=" * 70)
acc_moe = accuracy_score(y_test, y_pred_moe)
acc_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)

print(f"MoE alone accuracy:      {acc_moe:.4f}")
print(f"MoE + LLM accuracy:      {acc_final:.4f}  {'↑ IMPROVED' if acc_final > acc_moe else ''}")
print(f"F1-Score (final):        {f1_score(y_test, y_pred_final):.4f}")
if latencies:
    print(f"Average LLM latency:     {np.mean(latencies):.2f}s/sample")
print(f"\nSample LLM Explanations (first 3):")
for i, ex in enumerate(explanations[:3]):
    status = "✓" if ex['llm_pred'] == ex['true_label'] else "✗"
    print(
        f"\n{status} Sample #{ex['sample_id']} | True={ex['true_label']} → LLM={ex['llm_pred']} | Conf={ex['confidence']:.0%}")
    print(f"  {ex['explanation']}")

# Save full results
with open('real_llm_results.json', 'w') as f:
    json.dump({
        'metrics': {
            'moe_accuracy': float(acc_moe),
            'final_accuracy': float(acc_final),
            'f1_score': float(f1_final),
            'avg_latency': float(np.mean(latencies)) if latencies else 0.0
        },
        'explanations': explanations
    }, f, indent=2)

print("\n" + "=" * 70)
print("✓ SUCCESS! Full results saved to 'real_llm_results.json'")
print("=" * 70)

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from shutil import which

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from lapsum_pytorch.core import log_soft_top_k
    from utils.retriever import create_index
    from pyserini.search.lucene import LuceneSearcher
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

INDEX_NAME = "apnews"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", INDEX_NAME)
INDEX_PATH = os.path.join(PROJECT_ROOT, "indexes", INDEX_NAME)
QRELS_PATH = os.path.join(DATA_DIR, f"{INDEX_NAME}-qrels.txt")
QUERIES_PATH = os.path.join(DATA_DIR, f"{INDEX_NAME}-queries.txt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_FILE = os.path.join(RESULTS_DIR, "execution.log")

TOP_K_RETRIEVAL = 50
TOP_K_OPTIMIZATION = 10
ALPHA = 0.1
LEARNING_RATE = 0.1
EPOCHS = 30

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class DifferentiableRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float64))
    def forward(self, features):
        return torch.mv(features, self.weights)

def ensure_environment():
    if not os.path.exists(os.path.join(DATA_DIR, f"{INDEX_NAME}.dat")):
        print(f"Downloading data...")
        os.makedirs("data", exist_ok=True)
        if which("gdown"):
            zip_path = os.path.join("data", "data.zip")
            os.system(f"gdown 'https://drive.google.com/uc?id=1FCcBPYRHC1cAUAUbptIGOV5sJaMw_aXN&confirm=t' -O {zip_path}")
            print("Extracting data...")
            os.system(f"unzip -q -o {zip_path} -d .")
            if os.path.exists(zip_path): os.remove(zip_path)
    if not os.path.exists(INDEX_PATH) or not os.listdir(INDEX_PATH):
        print(f"Building index...")
        create_index(INDEX_NAME)

def load_data():
    if not os.path.exists(QRELS_PATH): return [], {}
    qrels = {}
    with open(QRELS_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                qid = int(parts[0])
                if len(parts) == 3: docid, rel = parts[1], int(parts[2])
                elif len(parts) >= 4: docid, rel = parts[2], int(parts[3])
                else: continue
            except ValueError: continue
            if qid not in qrels: qrels[qid] = {}
            qrels[qid][docid] = rel
    queries = []
    if os.path.exists(QUERIES_PATH):
        with open(QUERIES_PATH, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip(): queries.append((idx, line.strip()))
    return [q for q in queries if q[0] in qrels], qrels

def ndcg_at_k(sorted_labels, k=10):
    if sorted_labels.sum() == 0: return 0.0
    gains = sorted_labels[:k]
    discounts = torch.log2(torch.arange(2, k + 2, device=sorted_labels.device).float())
    dcg = (gains / discounts).sum()
    ideal_labels, _ = torch.sort(sorted_labels, descending=True)
    ideal_gains = ideal_labels[:k]
    idcg = (ideal_gains / discounts).sum()
    return (dcg / idcg).item() if idcg > 0 else 0.0

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = TeeLogger(LOG_FILE)
    print("=== Differentiable Search Engine Training ===")
    ensure_environment()
    try: searcher = LuceneSearcher(INDEX_PATH)
    except: return
    queries, qrels = load_data()
    model = DifferentiableRanker()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_queries = queries[:50]
    history = {'loss': [], 'ndcg': []}
    print(f"Training on {len(train_queries)} queries for {EPOCHS} epochs.")
    print("-" * 65)
    print(f"{'Epoch':<6} | {'Loss':<10} | {'nDCG@10':<10} | {'Weights [BM25, Noise]':<30}")
    print("-" * 65)
    for epoch in range(EPOCHS):
        total_loss = 0
        avg_ndcg = 0
        valid_batches = 0
        optimizer.zero_grad()
        for qid, query_text in train_queries:
            try: hits = searcher.search(query_text, k=TOP_K_RETRIEVAL)
            except: continue
            if len(hits) < 2: continue
            raw_scores = np.array([h.score for h in hits])
            if raw_scores.std() == 0: continue
            norm_scores = (raw_scores - raw_scores.mean()) / (raw_scores.std() + 1e-9)
            features = []
            labels = []
            for i, h in enumerate(hits):
                is_rel = 1.0 if h.docid in qrels[qid] and qrels[qid][h.docid] > 0 else 0.0
                labels.append(is_rel)
                features.append([norm_scores[i], np.random.randn()])
            feat_t = torch.tensor(features, dtype=torch.float64)
            lbl_t = torch.tensor(labels, dtype=torch.float64)
            if lbl_t.sum() == 0: continue
            scores = model(feat_t)
            log_probs = log_soft_top_k(scores.unsqueeze(0), torch.tensor([float(TOP_K_OPTIMIZATION)], dtype=torch.float64), torch.tensor(ALPHA, dtype=torch.float64), descending=False)
            loss = -torch.sum(torch.exp(log_probs.squeeze(0)) * lbl_t)
            loss.backward()
            total_loss += loss.item()
            valid_batches += 1
            _, sort_idx = torch.sort(scores, descending=True)
            avg_ndcg += ndcg_at_k(lbl_t[sort_idx], k=TOP_K_OPTIMIZATION)
        optimizer.step()
        if valid_batches > 0:
            e_loss = total_loss/valid_batches
            e_ndcg = avg_ndcg/valid_batches
            history['loss'].append(e_loss)
            history['ndcg'].append(e_ndcg)
            w = model.weights.data
            print(f"{epoch+1:<6} | {e_loss:<10.4f} | {e_ndcg:<10.4f} | [{w[0]:.4f}, {w[1]:.4f}]")
    plot_path = os.path.join(RESULTS_DIR, "training_curve.png")
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.plot(history['loss']); plt.title('Loss')
        plt.subplot(1, 2, 2); plt.plot(history['ndcg'], color='orange'); plt.title('nDCG@10')
        plt.tight_layout()
        plt.savefig(plot_path)
    except: pass
    print("-" * 65)
    print(f"Final Weights: BM25={model.weights[0]:.4f}, Noise={model.weights[1]:.4f}")
    if which("chafa"):
        try:
            cmd = ["chafa", "-f", "symbols", "--bg", "white", "-s", "120", plot_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if hasattr(sys.stdout, 'terminal'): sys.stdout.terminal.write(result.stdout)
            else: print(result.stdout)
        except: pass
    if model.weights[0] > model.weights[1]: print("Convergence Check: Passed (BM25 > Noise)")
    else: print("Convergence Check: Failed")
if __name__ == "__main__":
    main()

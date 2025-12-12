import torch
import sys
import os
import numpy as np

# --- PATH SETUP ---
# 1. Get the directory containing this script (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (Course_Project_deliverables/)
project_root = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path so we can import 'lapsum_pytorch'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Log file path setup
LOG_FILENAME = os.path.join(project_root, "results", "verification_log.txt")

# Ensure results directory exists
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

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

# Redirect stdout to capture results
sys.stdout = TeeLogger(LOG_FILENAME)

def run_single_case(alpha_value, check_probs=False, input_scores=None, k_target=None, backward_target_idx=3):
    try:
        # --- IMPORT FIX ---
        # Changed from 'from lapsum' to 'from lapsum_pytorch' to match new folder name
        from lapsum_pytorch import log_soft_top_k
    except ImportError as e:
        print(f"Import Error: {e}")
        print(f"Current Sys.Path: {sys.path}")
        sys.exit(1)

    torch.manual_seed(42)
    
    if input_scores is None:
        scores = torch.tensor([[10.0, 1.0, 8.0, 2.0, 9.0]], dtype=torch.float64, requires_grad=True)
    else:
        scores = input_scores.clone().detach().to(dtype=torch.float64).requires_grad_(True)
        
    if k_target is None:
        k_val = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    else:
        k_val = torch.tensor([float(k_target)], dtype=torch.float64, requires_grad=True)
        
    alpha = torch.tensor(float(alpha_value), dtype=torch.float64, requires_grad=True)
    descending = False

    print(f"\n==========================================")
    print(f" EXP: Alpha={alpha.item()} | Dim={scores.shape[1]} | K={k_val.item()}")
    print(f"==========================================")
    print(f"Input Scores: {scores.detach().numpy().tolist()}")

    try:
        log_output = log_soft_top_k(scores, k_val, alpha, descending=descending)
    except Exception as e:
        print(f"Execution Error: {e}")
        sys.exit(1)

    out_log_probs = log_output.detach().numpy()[0]
    out_probs = torch.exp(log_output).detach().numpy()[0]
    
    print("\n[Forward] Output Probabilities:")
    print(f"{'Idx':<5} | {'Input Score':<12} | {'LogProb':<12} | {'Probability':<12}")
    print("-" * 50)
    for i, (log_p, p) in enumerate(zip(out_log_probs, out_probs)):
        print(f"{i:<5} | {scores[0][i].item():<12.4f} | {log_p:<12.4f} | {p:<12.4f}")

    if check_probs:
        expected_high = [0, 2, 4] 
        expected_low = [1, 3]     
        for idx in expected_high:
            if out_log_probs[idx] <= -0.1:
                print(f"FAIL: Index {idx} should be selected but has Probability {out_probs[idx]:.4f}")
                sys.exit(1)     
        for idx in expected_low:
            if out_log_probs[idx] >= -2.0:
                print(f"FAIL: Index {idx} should be rejected but has Probability {out_probs[idx]:.4f}")
                sys.exit(1)

    target_idx = backward_target_idx
    loss = -log_output[0, target_idx]
    loss.backward()
    
    score_grads = scores.grad.detach().numpy()[0]
    target_grad = score_grads[target_idx]
    
    print("\n[Backward] Computed Gradients:")
    print(f"Target Index: {target_idx} (Score {scores[0][target_idx].item()})")
    print(f"Gradients w.r.t Scores:\n{score_grads}")
    print(f"Target Gradient: {target_grad:.6f}")

    if target_grad >= 0:
        print(f"FAIL: Target gradient is non-negative ({target_grad}). Model not learning.")
        sys.exit(1)
    
    if k_val.grad is None or alpha.grad is None:
        print("FAIL: Hyperparameter gradients missing.")
        sys.exit(1)
        
    print("\n[Hyperparameters] Gradients:")
    print(f"dLoss/dK: {k_val.grad.item():.6f}")
    print(f"dLoss/dAlpha: {alpha.grad.item():.6f}")

def run_tests():
    run_single_case(0.1, check_probs=True)
    run_single_case(1.0, check_probs=False)
    run_single_case(10.0, check_probs=False)
    
    large_scores = torch.tensor([[12.0, 1.0, 8.0, 2.0, 9.0, 5.0, 11.0, 3.0, 7.0, 4.0]])
    run_single_case(1.0, check_probs=False, input_scores=large_scores, k_target=4, backward_target_idx=5)

if __name__ == "__main__":
    run_tests()

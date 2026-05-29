"""Is it a SCALE bug (same shape, different magnitude) or a DIRECTION bug (different keys high)?
Compare HF relative scores (log of attn weights) vs Overfit raw scaled scores, layer 0 head 8."""
import numpy as np

hf = np.load(r"D:\Overfit\hf_attn.npy")        # [layers, heads, kv] weights
w = hf[0, 8].astype(np.float64)                # layer 0, head 8
hf_score_rel = np.log(np.clip(w, 1e-20, None))
hf_score_rel -= hf_score_rel.max()             # relative to max

ov = np.array([float(x) for x in open(r"D:\Overfit\overfit_scores_L0H8.txt").read().split()])
ov_rel = ov - ov.max()

print("key | HF rel-score | OV rel-score | (HF weight)")
order = np.argsort(hf_score_rel)[::-1][:12]    # top-12 by HF
for k in order:
    print(f"  {int(k):2d} | {hf_score_rel[k]:8.3f} | {ov_rel[k]:8.3f} | {w[k]:.4f}")

print(f"\nHF rel-score span: {hf_score_rel.min():.2f}..0   OV rel-score span: {ov_rel.min():.2f}..0")
print(f"ratio of spans (OV/HF): {ov_rel.min()/hf_score_rel.min():.2f}")
# correlation of shapes
print(f"corr(HF, OV) rel-scores: {np.corrcoef(hf_score_rel, ov_rel)[0,1]:.4f}")

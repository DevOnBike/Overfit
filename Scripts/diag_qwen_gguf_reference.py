"""Diagnostic (task #95): HF reference loading the SAME 3B GGUF Overfit uses (transformers
dequantizes it), so the comparison is apples-to-apples (same weights, same model)."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mp = r"C:\qwen3b"
gguf = "qwen.q4km.gguf"
tok = AutoTokenizer.from_pretrained(mp, gguf_file=gguf)
model = AutoModelForCausalLM.from_pretrained(mp, gguf_file=gguf, dtype=torch.float32, attn_implementation="eager")
model.eval()
print("HF-from-GGUF hidden:", model.config.hidden_size, "layers:", model.config.num_hidden_layers)

chatml = (
    "<|im_start|>system\nYou are a concise, helpful assistant running locally inside a .NET "
    "process. Answer only from context the user provides; if you are unsure, say so.<|im_end|>\n"
    "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = tok(chatml, return_tensors="pt").input_ids
print("prompt len:", ids.shape[1])

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=16, do_sample=False)
print("HF(3B-GGUF) GREEDY:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))

with torch.no_grad():
    o = model(ids, output_hidden_states=True, output_attentions=True)
logits = o.logits[0, -1].float()

# Save attention distributions (last query position) for all layers: [layers, heads, kv].
attn = np.stack([o.attentions[L][0, :, -1, :].float().numpy() for L in range(len(o.attentions))])
np.save(r"D:\Overfit\hf_attn.npy", attn)
print("saved hf_attn.npy", attn.shape)

# Save per-layer last-position hidden for cosine compare with Overfit (do this FIRST).
hs = o.hidden_states
arr = np.stack([hs[k][0, 0].float().numpy() for k in range(1, len(hs))])  # POS 0, [layers, dim]
np.save(r"D:\Overfit\hf_hidden.npy", arr)
print("saved hf_hidden.npy (POS 0)", arr.shape)

top = torch.topk(logits, 6)
print("top-6 first-token ids:", top.indices.tolist(), "vals:", [round(v.item(), 3) for v in top.values])

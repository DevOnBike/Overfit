"""Diagnostic (task #95): reference greedy generation from the HF Qwen2.5-3B on the EXACT
prompt that makes Overfit produce space-less garbage. Determines whether the fault is in
Overfit's forward pass (HF clean) or genuine model behaviour (HF also garbles)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mp = r"C:\qwen3b"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32)
model.eval()

chatml = (
    "<|im_start|>system\nYou are a concise, helpful assistant running locally inside a .NET "
    "process. Answer only from context the user provides; if you are unsure, say so.<|im_end|>\n"
    "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = tok(chatml, return_tensors="pt").input_ids
print("prompt len:", ids.shape[1])
print("first/last ids:", ids[0, :6].tolist(), "...", ids[0, -4:].tolist())

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=16, do_sample=False)
gen = out[0][ids.shape[1]:]
print("HF GREEDY OUTPUT:", repr(tok.decode(gen, skip_special_tokens=True)))
print("HF GREEDY token ids:", gen.tolist())

with torch.no_grad():
    out = model(ids, output_hidden_states=True)
logits = out.logits[0, -1].float()
top = torch.topk(logits, 8)
print("top-8 first-token logits:", [(repr(tok.decode([i.item()])), round(v.item(), 4)) for v, i in zip(top.values, top.indices)])

# Per-layer last-position hidden: L2 norm + max|.|  (hidden_states[0]=embeddings, [k]=after layer k-1)
hs = out.hidden_states
print("HF per-layer (lastpos max | ALLpos max | argmax pos | argmax dim):")
for k in range(1, len(hs)):
    layer = hs[k][0].float()                 # [seq, dim]
    lastmax = layer[-1].abs().max().item()
    allabs = layer.abs()
    allmax = allabs.max().item()
    pos = allabs.max(dim=1).values.argmax().item()   # which position has the biggest activation
    dim = allabs[pos].argmax().item()
    print(f"  layer {k-1:2d}: lastpos={lastmax:8.2f}  allpos={allmax:10.2f}  @pos={pos:3d} dim={dim}")

"""Parity reference for Bielik: HF transformers loads the SAME fp16 GGUF Overfit uses
(transformers dequantizes it), greedy-decodes a raw continuation prompt, and prints the
prompt token ids + the generated token ids. Compare against the Overfit side
(BielikParityTests). Same weights, same tokenizer (from the GGUF), greedy/temp=0 →
the continuations should match token-for-token if Overfit's forward pass is correct.

Run:  python Scripts/diag_bielik_parity.py
Needs: transformers, gguf, torch, accelerate  (pip install transformers gguf accelerate torch)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"C:\bielik"
GGUF = "Bielik-4.5B-v3.0-Instruct-fp16.gguf"
PROMPT = "Stolicą Polski jest"   # raw continuation — isolates the forward pass (no chat template)
N = 30

tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, gguf_file=GGUF, dtype=torch.float32, attn_implementation="eager"
).eval()
print("arch hidden:", model.config.hidden_size, "layers:", model.config.num_hidden_layers)

ids = tok(PROMPT, return_tensors="pt").input_ids
print("PROMPT_IDS", ids[0].tolist())

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
gen = out[0][ids.shape[1]:].tolist()
print("GEN_IDS", gen)
print("GEN_TEXT", repr(tok.decode(gen, skip_special_tokens=True)))

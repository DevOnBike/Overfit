"""Parity reference for Bielik on the REAL safetensors weights (the source of truth, not a
GGUF re-quant). HF transformers loads C:\\bielik-st, greedy-decodes a raw continuation prompt,
and writes {prompt_ids, gen_ids, gen_text} to bielik_st_ref.json. BielikSafetensorsParityTests
then loads the SAME safetensors via Overfit's SafetensorsLlamaLoader, feeds the SAME prompt_ids,
greedy-decodes, and compares token-for-token — a true forward-pass parity on identical weights.

Run:  python Scripts/diag_bielik_safetensors_parity.py
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"C:\bielik-st"
PROMPT = "Stolicą Polski jest"
N = 30

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
).eval()
print("arch hidden:", model.config.hidden_size, "layers:", model.config.num_hidden_layers,
      "dtype(src):", model.config.torch_dtype)

ids = tok(PROMPT, return_tensors="pt").input_ids
prompt_ids = ids[0].tolist()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
gen_ids = out[0][ids.shape[1]:].tolist()

try:
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
except Exception:
    gen_text = "<decode error>"

print("PROMPT_IDS", prompt_ids)
print("GEN_IDS", gen_ids)
print("GEN_TEXT", repr(gen_text.encode("ascii", "replace").decode()))

with open(r"D:\Overfit\bielik_st_ref.json", "w", encoding="utf-8") as f:
    json.dump({"prompt_ids": prompt_ids, "gen_ids": gen_ids, "gen_text": gen_text}, f)
print("wrote bielik_st_ref.json")

Check what tensors are in a GGUF file. Usage python3 check_gguf.py path_to_gguf
import sys
sys.path.insert(0, 'Scripts')
from convert_gguf import _read_gguf

if len(sys.argv) != 2
    print(Usage python3 check_gguf.py path_to_gguf)
    sys.exit(1)

print(fReading {sys.argv[1]} ...)
meta, tensors = _read_gguf(sys.argv[1])

print(fTotal tensors {len(tensors)})
print(fHas lm_head.weight            {'lm_head.weight' in tensors})
print(fHas model.embed_tokens.weight {'model.embed_tokens.weight' in tensors})
print()
print(Tensors with 'head', 'output', or 'embed' in name)
for name in sorted(tensors.keys())
    n = name.lower()
    if 'head' in n or 'output' in n or 'embed' in n
        shape = tensors[name].shape if hasattr(tensors[name], 'shape') else ''
        print(f  {name}  shape={shape})
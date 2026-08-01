# `LanguageModels/Tokenizers` — text to token ids

One implementation per vocabulary format the loaders can encounter: `HuggingFaceBpeTokenizer` for
`tokenizer.json`, `GgufEmbeddedTokenizer` / `GgufTokenizer` for vocabularies carried inside a GGUF
file, `QwenTokenizer` and `QwenChatTokenizer` for the Qwen family, `WordPieceTokenizer` for BERT-style
encoders, and `ByteLevelAlphabet` for the byte-level mapping BPE variants share.

## Why the embedded path matters

A GGUF file carries its own vocabulary, so a model downloaded as a single file needs no sidecar and no
Python conversion step. That is the difference between "download one file and run" and "install a
toolchain first", and it is the reason the GGUF path is the default in demos.

## Correctness is checked against the reference, byte for byte

Tokenisation that is *almost* right is worse than tokenisation that fails: the model still produces
fluent text, subtly off. The GPT-2 path is pinned to byte-parity against the reference implementation
rather than to a plausible-looking sample.

## The parser is a hostile-input surface

`tokenizer.json` and GGUF vocabularies are files from the internet, and unbounded recursion or an
unbounded loop while parsing one takes the host down. The `OVERFIT022`/`OVERFIT023` analyzer rules
(recursion and `while (true)`) are errors across this assembly and found real instances of exactly
that here — a bound has to be named at each site rather than assumed.

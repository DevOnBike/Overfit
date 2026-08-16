# `LanguageModels/Chat` — multi-turn conversation

Turns a token-level engine into a chat model: `ChatTemplate` renders messages into whatever prompt
format the model expects, `ChatSession` holds the conversation and its KV state, `StopSequenceDetector`
ends a turn at the right token, and `HuggingFaceChatModel` / `QwenChatModel` are the per-family
bindings.

Streaming, multi-turn history and a sliding context window are all implemented. `../Memory`
(`SummarizingChatSession`, `ChatHistoryCompactor`) is the alternative to dropping the oldest turns when
the window fills.

## What this is for

Embeddability, not raw speed: an in-process .NET chat engine with no server, no Python and no network
hop. A desktop application, a background service or a WPF demo can hold a `ChatSession` as a field.
That property is why the template is rendered here rather than by the caller — a chat format mismatch
produces answers that are fluent and ignore the system prompt, with nothing in the logs.

`ChatTemplateFormat` enumerates the formats; a model whose template is embedded in its GGUF metadata is
read by `HuggingFaceChatTemplate` in `../Loading`.

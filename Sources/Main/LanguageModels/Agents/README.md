# `LanguageModels/Agents` — tool-using loops

Agent control flow on top of `ChatSession`: `ReActAgent` runs the reason/act loop with tool calls
constrained by `ToolCallConstraint` (see `../Tools` and `../Constraints`) and an automatic `finish`
tool so a run always terminates. `CriticLoop` is the generate-critique-revise variant.
`CircuitBreaker` bounds a loop that is not converging, because an agent that retries forever is a
resource leak with a plausible explanation.

## Honest status

The loop is unit-tested and correct. **End to end on a 3B model it is unreliable**, and the failure is
specific: small constrained models produce well-formed tool calls whose JSON *string values* are wrong.
The grammar guarantees structure and has nothing to say about content, so this needs 7B+ to be useful.

That is a limitation of the model tier, not of the loop, and it is written down here so the next
person does not spend a day tuning prompts against it.

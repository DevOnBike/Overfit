---
name: asd-ste-100
description: Speak to the operator in ASD-STE-100 Simplified Technical English — approved words, short sentences, active voice, one instruction per sentence. Applies to prose only, never to code, file contents, commands, or quoted output.
---

# Write to the operator in ASD-STE-100 (Simplified Technical English)

ASD-STE-100 is the controlled language of aerospace maintenance documentation. Write all prose that the
operator reads in that language.

## What this applies to, and what it does not

Apply these rules to your prose: explanations, findings, recommendations, and status.

Do **not** apply them to:

- code, file contents, and file paths;
- commands, command output, log lines, and error messages you quote;
- names of symbols, packages, tests, and tasks;
- text you write into a file, a commit message, or a document.

Quote a measured value or an error message exactly. Do not simplify a quotation.

## Words

- Use one word for one meaning. Do not use a synonym for a thing you named before.
- Use one word as one part of speech. If you use `test` as a noun, do not also use it as a verb.
- Use the technical name of a thing. Do not replace it with a general word.
- Do not use slang, idioms, or jargon.
- Do not use a word to make the text pleasant. Every word must give information.

## Sentences

- Write a maximum of 20 words in an instruction.
- Write a maximum of 25 words in a description.
- Write one instruction in one sentence.
- Use the active voice. Write *the test found the defect*, not *the defect was found*.
- Use simple tenses: simple present, simple past, simple future.
- Do not use the `-ing` form as a noun.
- Use `the` and `a` where they are possible. Do not remove words to make a sentence short.
- Write a maximum of 6 sentences in a paragraph.
- Write one topic in one paragraph.

## Order

- Give the warning before the step it applies to. Never after.
- Give the most important information first.
- Write a sequence of steps as a numbered list.
- Write a set of conditions as a bullet list.

## What must not change

These rules control **how** you write. They do not control **what** you write.

- Report a measurement with its number and its unit.
- Report what you did not check. Say it directly: *I did not check X.*
- Report a result that disagrees with your conclusion.
- Do not remove a limitation to make a sentence shorter.

A short sentence that hides a limitation is a failure of this style, not an application of it.

**This repository already records what that failure costs.** A test that hid its own emptiness passed for
months. A comment that hid a deleted type was quoted as a design precedent. Brevity that removes evidence
is the defect this file must not introduce.

## Why this file is in the repository and gitignored

`.gitignore` excludes `.claude/` completely. This file is therefore invisible to git until somebody runs
`git add -f`. That is the same trap that deleted the first version of `Scripts/lab.py`, which `CLAUDE.md`
records. If you copy this style to another repository, force-add it, or it disappears at the next clone.

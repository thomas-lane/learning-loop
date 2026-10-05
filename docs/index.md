# Documentation index

Every document in this repository, by what you want to do. Read the Markdown directly, or browse it
rendered, with working links and diagrams, with `uv run loop docs --open`.

## Start here

| I want to... | Read |
|---|---|
| know what the project does, its status, and how to set it up and run it | [README](../README.md) |
| see how the pieces fit together: components, machines, module map, data flow, who sees what | [Architecture](architecture.md) |
| understand the research method: cycle protocol, controls, editing, acceptance, cost accounting, training, metrics | [Method](experiment.md) |
| look up a term | [Glossary](glossary.md) |

## Running experiments

| I want to... | Read |
|---|---|
| know what a `loop` command does and its flags | [CLI reference](cli.md) (generated from the parser) |
| write or change an experiment, machine or model profile | [Configuration reference](configuration.md) (generated from the schemas, plus rules that span several fields) |
| run a real experiment step by step, or recover from an error | [Operations](operations.md) |
| use a Runpod (or other SSH) GPU pod for serving and training, or serve with vLLM | [Runpod deployment](runpod.md) |
| find a file in a run directory and know what it contains | [Run layout](run-layout.md) |

## Tasks and the agent

| I want to... | Read |
|---|---|
| add or change a task, generator or split; see what the learner sees and how grading is kept hidden from it | [Tasks and agent](../evaluation/README.md) |

## Developing

| I want to... | Read |
|---|---|
| change code: commands, tests, integrity invariants, where things live | [Developer guide (AGENTS.md)](../AGENTS.md) |
| know which document to update when the code changes | [Developer guide: Keeping documentation current](../AGENTS.md#keeping-documentation-current) |

# Documentation index

All documentation for this repository, by what you want to do. Browse it locally with
`uv run loop docs --open` (rendered pages, working links, diagrams), or read the Markdown
directly.

## Start here

| I want to... | Read |
|---|---|
| understand what the project does and run it | [README](../README.md): purpose, status, setup, running, smoke tests |
| understand how the pieces fit together | [Architecture](architecture.md): components, machine roles, module map, data flow, who sees what |
| understand the research method | [Method](experiment.md): cycle protocol, controls, editing rules, acceptance, cost accounting, training, metrics |
| look up a term | [Glossary](glossary.md) |

## Running experiments

| I want to... | Read |
|---|---|
| know what a `loop` command does and its flags | [CLI reference](cli.md) (generated from the parser) |
| write or change an experiment, machine or model profile | [Configuration reference](configuration.md) (generated from the schemas) + cross-field rules |
| run a real experiment end to end, or recover from an error | [Operations](operations.md): ordered protocol, smoke path, troubleshooting |
| use a Runpod (or other SSH) GPU pod for serving and training | [Runpod deployment](runpod.md): pod choice, setup, running, costs, troubleshooting |
| find a file in a run directory or know what it contains | [Run layout](run-layout.md) |

## Tasks and the agent

| I want to... | Read |
|---|---|
| add or change a task, generator or split | [Tasks and agent](../evaluation/README.md): task contract, replay contract, generators, splits |
| know exactly what the learner sees and how grading is isolated | [Tasks and agent](../evaluation/README.md): the agent, outputs, how a trial runs |

## Developing

| I want to... | Read |
|---|---|
| change code: commands, tests, invariants, where things live | [Developer guide (AGENTS.md)](../AGENTS.md) |
| keep the documentation current | [Developer guide: Keeping documentation current](../AGENTS.md#keeping-documentation-current) |

## All documents

| Document | Owns |
|---|---|
| [README.md](../README.md) | setup, operation, component status |
| [docs/index.md](index.md) | this index |
| [docs/architecture.md](architecture.md) | components, roles, repository and module layout, data flow, isolation, identities, extension points |
| [docs/experiment.md](experiment.md) | method, protocol, controls, acceptance, cost accounting, metrics |
| [docs/run-layout.md](run-layout.md) | run directory contents and mutability |
| [docs/cli.md](cli.md) | `loop` commands, arguments, exit codes, workflows |
| [docs/configuration.md](configuration.md) | YAML keys, loading rules, cross-field rules |
| [docs/operations.md](operations.md) | experiment protocol, smoke path, troubleshooting |
| [docs/runpod.md](runpod.md) | Runpod / SSH GPU host deployment |
| [docs/glossary.md](glossary.md) | terms |
| [evaluation/README.md](../evaluation/README.md) | tasks, generators, splits, agent, grading, replay contract |
| [AGENTS.md](../AGENTS.md) | development commands, tests, integrity invariants, documentation policy |

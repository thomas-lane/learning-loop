"""Generated documentation stays in sync with the CLI parser and the config schemas.

If this fails after changing a command, an argument or a config field: run `uv run loop docs-gen`
and commit the regenerated docs/cli.md and docs/configuration.md.
"""

from learning_loop.cli import main
from learning_loop.docs_tools import docgen


def test_generated_docs_are_current():
    stale = docgen.check_docs()
    assert not stale, f"regenerate with `uv run loop docs-gen`: {[str(p) for p in stale]}"


def test_every_config_field_and_cli_argument_is_described():
    assert docgen.missing_descriptions() == []


def test_docs_check_command(capsys):
    assert main(["docs-gen", "--check"]) == 0


def test_markdown_placeholders_become_code_spans():
    assert docgen.md("see runs/<run-id>/ now") == "see `runs/<run-id>/` now"
    assert docgen.md("default: <experiment>-<UTC-stamp>.") == "default: `<experiment>-<UTC-stamp>`."
    assert docgen.md("already `runs/<id>` fine") == "already `runs/<id>` fine"

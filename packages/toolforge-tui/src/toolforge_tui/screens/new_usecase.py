"""Modal screen for creating a new use case."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, TextArea


class NewUsecaseScreen(ModalScreen[tuple[str, str] | None]):
    CSS = """\
    NewUsecaseScreen {
        align: center middle;
    }
    #dialog {
        width: 64;
        height: auto;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    #dialog Label {
        margin-top: 1;
    }
    #id-input {
        margin-bottom: 0;
    }
    #prompt-area {
        height: 8;
    }
    #buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    #buttons Button {
        margin-left: 1;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("[bold]New Use Case[/]", markup=True)
            yield Label("ID (letters, digits, underscores):")
            yield Input(id="id-input", placeholder="e.g. invoice_extraction")
            yield Label("Prompt (describe what the tools should do):")
            yield TextArea(id="prompt-area")
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Create", variant="primary", id="create")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        usecase_id = self.query_one("#id-input", Input).value.strip()
        prompt = self.query_one("#prompt-area", TextArea).text.strip()
        if not usecase_id or not prompt:
            self.notify("Both ID and prompt are required.", severity="warning")
            return
        self.dismiss((usecase_id, prompt))

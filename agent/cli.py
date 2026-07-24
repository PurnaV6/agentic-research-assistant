from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.orchestrator import run_agent

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def research(
    question: str = typer.Argument(..., help="The research question to investigate"),
    max_steps: int = typer.Option(8, help="Maximum agent reasoning/tool-call steps"),
    output: str = typer.Option("report.md", help="File to write the final report to"),
):
    """Run the research agent on QUESTION and write a cited markdown report."""
    console.print(f"[bold cyan]Researching:[/bold cyan] {question}\n")

    def _print_step(entry: dict) -> None:
        if entry["type"] == "tool_call":
            console.print(
                Panel(
                    f"args: {entry['args']}\nresult: {str(entry['result'])[:400]}",
                    title=f"tool call -> {entry['name']}",
                    border_style="yellow",
                )
            )
        else:
            console.print(Panel("(final answer)", border_style="green"))

    result = run_agent(question, max_steps=max_steps)
    for entry in result["trace"]:
        _print_step(entry)

    console.print("\n[bold green]Final report:[/bold green]\n")
    console.print(Markdown(result["report"]))

    Path(output).write_text(result["report"])
    console.print(f"\n[dim]Saved to {output} ({result['steps']} steps, {len(result['sources'])} sources)[/dim]")


if __name__ == "__main__":
    app()

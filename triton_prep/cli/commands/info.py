import typer

from triton_prep.services.inspector import ModelInspector

cli = typer.Typer(help="Display information about a HuggingFace model.")


@cli.command()
def show(
    model_id: str = typer.Argument(..., help="HuggingFace model id or path."),
) -> None:
    """
    Displays high-level properties of a HuggingFace model.
    """
    inspector = ModelInspector()
    info = inspector.inspect(model_id)

    typer.echo(f"Model: {info.model_id}")
    typer.echo(f"Task type: {info.task_type}")
    typer.echo(f"Parameters: {info.num_parameters}")
    typer.echo(f"Embedding dim: {info.embedding_dim}")
    typer.echo(f"Vocab size: {info.vocab_size}")

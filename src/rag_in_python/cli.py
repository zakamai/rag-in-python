"""
Command-line interface for the RAG system.
"""

from typing import Optional, List
from pathlib import Path
import logging
import os

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .core import RAGSystem

# Set up rich console
console = Console()
app = typer.Typer(help="RAG in Python - A comprehensive RAG toolkit")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def index(
    data_dir: Path = typer.Argument(
        ..., 
        help="Directory containing documents to index"
    ),
    index_path: Path = typer.Option(
        Path("./vector_index"), 
        "--index-path", "-i",
        help="Path to save the vector index"
    ),
    batch_size: int = typer.Option(
        100, 
        "--batch-size", "-b",
        help="Batch size for indexing"
    ),
) -> None:
    """Index documents from a directory."""
    
    if not data_dir.exists():
        console.print(f"[red]Error: Directory {data_dir} does not exist[/red]")
        raise typer.Exit(1)
        
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
        
    console.print(f"[blue]Indexing documents from {data_dir}[/blue]")
    
    # Initialize RAG system
    rag_system = RAGSystem(vector_store_path=index_path)
    
    # Get all files in directory
    file_paths = list(data_dir.rglob("*"))
    file_paths = [p for p in file_paths if p.is_file()]
    
    if not file_paths:
        console.print(f"[yellow]No files found in {data_dir}[/yellow]")
        return
        
    console.print(f"Found {len(file_paths)} files to index")
    
    # Index documents
    with console.status(f"[bold green]Indexing {len(file_paths)} files..."):
        rag_system.load_and_index_files(file_paths, batch_size=batch_size)
        
    # Save index
    rag_system.save_index()
    
    console.print(f"[green]✓ Successfully indexed documents to {index_path}[/green]")


@app.command()
def query(
    question: str = typer.Argument(
        ..., 
        help="Question to ask"
    ),
    index_path: Path = typer.Option(
        Path("./vector_index"), 
        "--index-path", "-i",
        help="Path to the vector index"
    ),
    top_k: int = typer.Option(
        5, 
        "--top-k", "-k",
        help="Number of documents to retrieve"
    ),
    with_citations: bool = typer.Option(
        False, 
        "--citations", "-c",
        help="Include citations in the response"
    ),
) -> None:
    """Query the indexed documents."""
    
    if not index_path.exists():
        console.print(f"[red]Error: Index path {index_path} does not exist[/red]")
        console.print("Run 'rag-cli index' first to create an index")
        raise typer.Exit(1)
        
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
        
    console.print(f"[blue]Question:[/blue] {question}")
    console.print()
    
    # Initialize RAG system and load index
    rag_system = RAGSystem(vector_store_path=index_path)
    
    with console.status("[bold green]Loading index..."):
        rag_system.load_index()
        
    with console.status(f"[bold green]Searching {top_k} relevant documents..."):
        result = rag_system.query(question, top_k=top_k)
        
    # Display response
    console.print(f"[green]Answer:[/green]")
    console.print(result["response"])
    console.print()
    
    # Display metadata
    console.print(f"[dim]Retrieved {result['retrieved_documents']} relevant documents[/dim]")
    
    # Display citations if requested
    if with_citations and result.get("sources"):
        console.print("\n[blue]Sources:[/blue]")
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Score", style="yellow", width=8)
        
        for i, source in enumerate(result["sources"], 1):
            score = source.get("score", "N/A")
            source_name = source.get("filename", source.get("source", "Unknown"))
            if isinstance(score, float):
                score = f"{score:.3f}"
            table.add_row(str(i), str(source_name), str(score))
            
        console.print(table)


@app.command()
def interactive(
    index_path: Path = typer.Option(
        Path("./vector_index"), 
        "--index-path", "-i",
        help="Path to the vector index"
    ),
    top_k: int = typer.Option(
        5, 
        "--top-k", "-k",
        help="Number of documents to retrieve"
    ),
) -> None:
    """Start an interactive query session."""
    
    if not index_path.exists():
        console.print(f"[red]Error: Index path {index_path} does not exist[/red]")
        console.print("Run 'rag-cli index' first to create an index")
        raise typer.Exit(1)
        
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        raise typer.Exit(1)
        
    # Initialize RAG system and load index
    console.print("[blue]Loading RAG system...[/blue]")
    rag_system = RAGSystem(vector_store_path=index_path)
    
    with console.status("[bold green]Loading index..."):
        rag_system.load_index()
        
    console.print("[green]✓ RAG system ready![/green]")
    console.print("Type 'quit' or 'exit' to end the session")
    console.print("=" * 60)
    
    while True:
        try:
            question = typer.prompt("\n🤖 Ask a question")
            
            if question.lower() in ["quit", "exit", "q"]:
                console.print("[blue]Goodbye![/blue]")
                break
                
            if not question.strip():
                continue
                
            # Query the system
            result = rag_system.query(question, top_k=top_k)
            
            console.print(f"\n[green]Answer:[/green]")
            console.print(result["response"])
            console.print(f"\n[dim]Sources: {result['retrieved_documents']} documents[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[blue]Goodbye![/blue]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
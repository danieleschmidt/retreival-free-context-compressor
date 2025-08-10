"""Command-line interface for retrieval-free context compressor."""


def main():
    """Main CLI entry point."""
    from .plugins import CLIInterface

    cli = CLIInterface()
    cli.main()


if __name__ == "__main__":
    main()

"""Allow running the CLI as a module: python -m contextfs.cli"""

from . import app

if __name__ == "__main__":
    app()

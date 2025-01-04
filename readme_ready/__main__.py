"""Entry point for readme_ready."""

import os


def entry_point():
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["HF_TOKEN"] = "dummy"
    from .main import main

    main()


if __name__ == "__main__":  # pragma: no cover
    entry_point()

"""Main application entry point."""

from .config import DEFAULT_CONFIG


def main() -> None:
    print("Financial XAI project initialized.")
    print(f"Project root: {DEFAULT_CONFIG.project_root}")


if __name__ == "__main__":
    main()

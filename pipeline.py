"""Root-level compatibility entry point used by app.py's upload control."""

from crowdguard.pipeline import main, run_pipeline

__all__ = ["run_pipeline"]


if __name__ == "__main__":
    main()

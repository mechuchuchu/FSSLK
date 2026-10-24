"""FSSLK's first CPU-only distributed runtime prototype."""

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy so that ``import fsslk`` does not pull in torch.export machinery.
    if name == "wrap":
        from .graph import wrap

        return wrap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

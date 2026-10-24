"""FSSLK's CPU distributed runtime."""

__version__ = "0.1.0"
__all__ = ["__version__", "wrap", "fsslk_ddp"]


def __getattr__(name: str):
    # Lazy so that importing FSSLK does not load graph capture or RPC modules.
    if name == "wrap":
        from .graph import wrap

        return wrap
    if name == "fsslk_ddp":
        from .ddp import fsslk_ddp

        return fsslk_ddp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

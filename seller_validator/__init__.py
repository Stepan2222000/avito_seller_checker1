"""Пакет агента проверки продавцов Авито."""

__all__ = ["run"]


def run(*args, **kwargs):
    from .agent import run as _run

    return _run(*args, **kwargs)

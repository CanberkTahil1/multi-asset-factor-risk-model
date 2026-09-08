"""Loaders, cache, manifest and data contracts.

CLAUDE.md invariant 1: this is the ONLY package permitted to touch the network.
Everything in ``factors``, ``risk``, ``costs`` and ``backtest`` reads from the
cache. If one of them needs data, add a loader here -- never fetch inline.
"""

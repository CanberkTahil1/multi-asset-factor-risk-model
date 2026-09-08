"""Multi-asset factor risk model with a cost-aware implementation layer.

This project does not forecast returns. Expected returns enter only as a fixed,
documented input to the optimizer so that it has something to trade against.
See CLAUDE.md for the working constraints and SPEC.md for the methodology.
"""

from mafrm.config import SEED, Config, ConfigError, load

__all__ = ["SEED", "Config", "ConfigError", "__version__", "load"]

__version__ = "0.1.0"

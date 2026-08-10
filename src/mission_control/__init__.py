"""Mission control layer for an autonomous inspection multirotor.

The package is deliberately split so that the parts that must be *measured*
(estimation, control, safety) have no dependency on the parts that talk to a
vehicle (``mavlink_bridge``). Everything under ``sim`` is a reference plant used
to score controllers against each other; PX4 SITL is a second backend behind the
same interface.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

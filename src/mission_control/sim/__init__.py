"""Reference multirotor simulation.

This exists so the MPC-versus-PID result is reproducible on any machine in
seconds, with no Gazebo, no GPU and no network. It is a 6-DOF rigid-body model
with rotor dynamics, aerodynamic drag and wind, wrapped in the same cascaded
inner loop PX4 runs, so a tracker cannot tell the difference at its interface.

It is *not* a claim to be as faithful as Gazebo. See ``sitl/`` for the PX4 SITL
backend and ``docs/LIMITATIONS.md`` for what the simplification costs.
"""

from .inner_loop import InnerLoop, InnerLoopGains
from .quadrotor import Mixer, QuadrotorParams, QuadrotorPlant
from .sensors import GnssParams, GnssReceiver, ImuParams, ImuSensor
from .wind import WindField, WindParams

__all__ = [
    "Mixer",
    "QuadrotorParams",
    "QuadrotorPlant",
    "InnerLoop",
    "InnerLoopGains",
    "WindField",
    "WindParams",
    "GnssReceiver",
    "GnssParams",
    "ImuSensor",
    "ImuParams",
]

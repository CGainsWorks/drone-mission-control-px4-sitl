"""6-DOF quadrotor plant.

State vector, integrated with RK4 at a fixed step:

===========  ======  ===========================================
slice        size    meaning
===========  ======  ===========================================
``0:3``      3       position, NED, m
``3:6``      3       velocity, NED, m/s
``6:10``     4       orientation quaternion, body to NED
``10:13``    3       body angular velocity, rad/s
``13:17``    4       rotor thrusts, N (first-order motor lag)
===========  ======  ===========================================

Rotor thrusts are part of the state rather than an instantaneous function of the
command because motor lag is the dominant unmodelled delay in a multirotor and
it is exactly what punishes an aggressive controller. Leaving it out would
flatter the MPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..attitude import cross3, quat_multiply, quat_normalise, quat_to_rotation_matrix
from ..geodesy import GRAVITY
from ..types import VehicleState

STATE_SIZE = 17


@dataclass
class QuadrotorParams:
    """Physical parameters of a ~1.6 kg inspection quadrotor in X configuration.

    The numbers are representative of a 450-class airframe with 9-inch props.
    They are not measured from any specific vehicle; see the clean-room note in
    the README.
    """

    mass: float = 1.6
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.0170, 0.0170, 0.0310]))
    arm_length: float = 0.25
    #: Maximum static thrust per rotor, N. Four of these against a 15.7 N weight
    #: gives a thrust-to-weight ratio of 2.17, typical for an inspection frame
    #: carrying a gimbal.
    max_rotor_thrust: float = 8.5
    #: Rotor reaction torque per newton of thrust, m. Sets yaw authority.
    torque_per_thrust: float = 0.016
    #: First-order motor+propeller time constant, s.
    motor_tau: float = 0.035
    #: Linear drag, N per m/s, in body axes. Dominated by rotor drag at low speed.
    drag_linear: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.20, 0.10]))
    #: Quadratic drag, N per (m/s)^2, in body axes. Airframe form drag.
    drag_quadratic: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.10, 0.20]))
    #: Angular rate damping, N·m per rad/s.
    angular_damping: float = 0.010

    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def max_thrust(self) -> float:
        return 4.0 * self.max_rotor_thrust

    @property
    def hover_thrust(self) -> float:
        """Normalised collective thrust that holds a hover. Around 0.46 here."""
        return self.weight / self.max_thrust

    @property
    def thrust_to_weight(self) -> float:
        return self.max_thrust / self.weight

    @property
    def max_tilt_for_hover(self) -> float:
        """Tilt at which full thrust can no longer hold altitude, rad."""
        return float(np.arccos(np.clip(self.weight / self.max_thrust, -1.0, 1.0)))

    def rotor_geometry(self) -> tuple:
        """Rotor positions (body, m) and spin directions.

        X configuration, matching the PX4 ``quad_x`` airframe: rotor 0 front
        right, 1 rear left, 2 front left, 3 rear right. Directions are ``+1``
        for counter-clockwise seen from above, which reacts a ``+z_body``
        (nose-right yaw) torque onto the airframe.
        """
        d = self.arm_length / np.sqrt(2.0)
        positions = np.array(
            [
                [+d, +d, 0.0],  # front right
                [-d, -d, 0.0],  # rear left
                [+d, -d, 0.0],  # front left
                [-d, +d, 0.0],  # rear right
            ]
        )
        directions = np.array([+1.0, +1.0, -1.0, -1.0])
        return positions, directions

    def mixer_matrix(self) -> np.ndarray:
        """Maps rotor thrusts to ``[collective, tau_x, tau_y, tau_z]``."""
        positions, directions = self.rotor_geometry()
        rows = [
            np.ones(4),  # collective thrust magnitude
            -positions[:, 1],  # roll: thrust at -y rolls positive
            positions[:, 0],  # pitch: thrust at +x pitches nose up
            directions * self.torque_per_thrust,  # yaw: rotor reaction torque
        ]
        return np.vstack(rows)


class Mixer:
    """Control allocation: a wrench demand becomes four rotor thrusts.

    Shared by the plant and the inner loop so the saturation policy is defined
    in exactly one place. When the demand does not fit inside the rotors'
    range, something has to give, and **the order it gives in is the whole
    design**:

    1. **Roll and pitch** are satisfied first. They are what keeps the vehicle
       the right way up and what points the thrust vector, so they are the last
       thing to surrender.
    2. **Collective** next. Losing a little climb rate is survivable.
    3. **Yaw** last, with whatever authority is left over.

    Yaw goes last because on a multirotor it is by far the weakest axis — it
    comes from rotor drag torque, not from thrust differential across an arm, so
    it is an order of magnitude weaker than roll and pitch — and because heading
    is the one thing a multirotor can be wrong about for a second without
    consequence.

    Getting this order wrong is not a subtle bug. An earlier version scaled all
    three torque axes together, so the large yaw demand generated while turning
    a corner scaled roll and pitch down with it. The vehicle stopped being able
    to hold the turn, flew wide, demanded more correction, and departed. The
    symptom looked like a controller tuning problem and was not.
    """

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.matrix = params.mixer_matrix()
        self.inverse = np.linalg.inv(self.matrix)
        #: Per-rotor thrust contribution of a unit demand on each torque axis.
        #: Collective needs no such vector: it is an equal shift on all four,
        #: which the allocation applies directly.
        self._roll = self.inverse @ np.array([0.0, 1.0, 0.0, 0.0])
        self._pitch = self.inverse @ np.array([0.0, 0.0, 1.0, 0.0])
        self._yaw = self.inverse @ np.array([0.0, 0.0, 0.0, 1.0])

    def allocate(self, collective: float, torque: np.ndarray) -> np.ndarray:
        f_max = self.params.max_rotor_thrust
        torque = np.asarray(torque, dtype=float)

        # --- 1. roll and pitch --------------------------------------------
        thrusts = self._roll * torque[0] + self._pitch * torque[1]
        span = float(thrusts.max() - thrusts.min())
        if span > f_max:
            # Even alone they do not fit. Scale both together, which preserves
            # the *direction* of the demanded tilt rate even though it cannot
            # deliver the magnitude.
            thrusts *= f_max / span
        # Centre what we have inside the available band before adding anything.
        thrusts -= 0.5 * (thrusts.max() + thrusts.min() - f_max)

        # --- 2. collective ------------------------------------------------
        # Shift the whole set towards the requested collective, as far as the
        # rotors allow.
        centre = float(np.mean(thrusts))
        target = float(np.clip(collective / 4.0, 0.0, f_max))
        shift = float(np.clip(target - centre, -thrusts.min(), f_max - thrusts.max()))
        thrusts = thrusts + shift

        # --- 3. yaw, with whatever headroom is left ------------------------
        yaw = self._yaw * torque[2]
        if np.any(np.abs(yaw) > 1e-12):
            # Largest fraction of the yaw demand that keeps every rotor legal.
            scale = 1.0
            for value, delta in zip(thrusts, yaw):
                if delta > 1e-12:
                    scale = min(scale, (f_max - value) / delta)
                elif delta < -1e-12:
                    scale = min(scale, value / -delta)
            thrusts = thrusts + yaw * max(0.0, min(1.0, scale))

        return np.clip(thrusts, 0.0, f_max)

    def achieved(self, collective: float, torque: np.ndarray) -> np.ndarray:
        """The wrench actually produced after allocation and saturation.

        Used by the tests to check the priority order holds: roll and pitch must
        come out intact in cases where yaw does not.
        """
        return self.matrix @ self.allocate(collective, torque)


class QuadrotorPlant:
    """Rigid-body multirotor with rotor dynamics, drag and wind.

    The plant is deterministic given its initial state and command sequence; all
    randomness lives in :mod:`mission_control.sim.wind` and
    :mod:`mission_control.sim.sensors`, both seeded explicitly.
    """

    def __init__(self, params: QuadrotorParams = None, initial_state: VehicleState = None):
        self.params = params or QuadrotorParams()
        self.mixer = Mixer(self.params)
        # Rows 1..3 of the mixer map rotor thrusts to body torque. Reusing them
        # here instead of summing cross products per rotor is both faster — this
        # runs four times per RK4 step at 500 Hz — and stronger: the plant and
        # the control allocator cannot disagree about the vehicle's geometry,
        # because there is only one description of it.
        self._torque_matrix = self.mixer.matrix[1:4, :]
        self._drag_linear = self.params.drag_linear
        self._drag_quadratic = self.params.drag_quadratic
        self._inertia = self.params.inertia
        self._gravity_ned = np.array([0.0, 0.0, self.params.mass * GRAVITY])
        self.state = np.zeros(STATE_SIZE)
        self.state[6] = 1.0  # identity quaternion
        # Start with the rotors already producing hover thrust so that a run
        # beginning in the air does not sag for one motor time constant.
        self.state[13:17] = self.params.weight / 4.0
        if initial_state is not None:
            self.set_state(initial_state)
        self.wind_ned = np.zeros(3)
        self._last_rotor_command = np.full(4, self.params.weight / 4.0)

    # ------------------------------------------------------------------
    # state access
    # ------------------------------------------------------------------
    def set_state(self, state: VehicleState) -> None:
        self.state[0:3] = state.position
        self.state[3:6] = state.velocity
        self.state[6:10] = quat_normalise(state.orientation)
        self.state[10:13] = state.angular_velocity

    def vehicle_state(self, time_s: float = 0.0) -> VehicleState:
        return VehicleState(
            time_s=time_s,
            position=self.state[0:3].copy(),
            velocity=self.state[3:6].copy(),
            orientation=self.state[6:10].copy(),
            angular_velocity=self.state[10:13].copy(),
        )

    @property
    def rotor_thrusts(self) -> np.ndarray:
        return self.state[13:17].copy()

    @property
    def on_ground(self) -> bool:
        return self.state[2] > -0.02 and abs(self.state[5]) < 0.15

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _derivative(self, state: np.ndarray, rotor_command: np.ndarray) -> np.ndarray:
        p = self.params
        velocity = state[3:6]
        quat = quat_normalise(state[6:10])
        omega = state[10:13]
        thrusts = state[13:17]

        rot = quat_to_rotation_matrix(quat)

        # Thrust acts along -z_body, so it is the third column of the rotation
        # matrix scaled by the total, negated.
        force_ned = rot[:, 2] * -float(np.sum(thrusts)) + self._gravity_ned

        # Drag opposes airspeed, not ground speed. Evaluated in body axes
        # because a multirotor is not isotropic.
        airspeed_body = rot.T @ (velocity - self.wind_ned)
        drag_body = -(
            self._drag_linear * airspeed_body
            + self._drag_quadratic * airspeed_body * np.abs(airspeed_body)
        )
        force_ned = force_ned + rot @ drag_body

        # Attitude kinematics: q_dot = 0.5 * q ⊗ [0, omega].
        quat_dot = 0.5 * quat_multiply(quat, np.array([0.0, omega[0], omega[1], omega[2]]))

        # Euler's equation, with rotor thrust and reaction torques taken from
        # the allocation matrix and rate damping subtracted.
        torque = self._torque_matrix @ thrusts - p.angular_damping * omega
        inertia = self._inertia
        omega_dot = (torque - cross3(omega, inertia * omega)) / inertia

        derivative = np.empty(STATE_SIZE)
        derivative[0:3] = velocity
        derivative[3:6] = force_ned / p.mass
        derivative[6:10] = quat_dot
        derivative[10:13] = omega_dot
        derivative[13:17] = (rotor_command - thrusts) / p.motor_tau
        return derivative

    def step(self, rotor_command: np.ndarray, dt: float) -> None:
        """Advance the plant by ``dt`` with RK4."""
        rotor_command = np.clip(
            np.asarray(rotor_command, dtype=float), 0.0, self.params.max_rotor_thrust
        )
        self._last_rotor_command = rotor_command

        s = self.state
        k1 = self._derivative(s, rotor_command)
        k2 = self._derivative(s + 0.5 * dt * k1, rotor_command)
        k3 = self._derivative(s + 0.5 * dt * k2, rotor_command)
        k4 = self._derivative(s + dt * k3, rotor_command)
        s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        s[6:10] = quat_normalise(s[6:10])
        s[13:17] = np.clip(s[13:17], 0.0, self.params.max_rotor_thrust)

        # Ground contact. Deliberately crude: the vehicle cannot go below the
        # ground plane and cannot slide or bounce. Enough to make takeoff and
        # landing behave; not a contact model.
        if s[2] > 0.0:
            s[2] = 0.0
            if s[5] > 0.0:
                s[5] = 0.0
                s[3:5] *= 0.0
                s[10:13] *= 0.0

        self.state = s

    def specific_force_body(self) -> np.ndarray:
        """What an ideal accelerometer at the centre of mass would read, m/s^2.

        Specific force excludes gravity, so this is the plant's own answer to
        the question the IMU model then corrupts with noise and bias.
        """
        p = self.params
        rot = quat_to_rotation_matrix(quat_normalise(self.state[6:10]))
        force_body = np.array([0.0, 0.0, -float(np.sum(self.state[13:17]))])

        airspeed_body = rot.T @ (self.state[3:6] - self.wind_ned)
        force_body -= (
            self._drag_linear * airspeed_body
            + self._drag_quadratic * airspeed_body * np.abs(airspeed_body)
        )
        return force_body / p.mass

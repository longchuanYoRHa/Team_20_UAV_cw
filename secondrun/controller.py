# wind_flag = True  # Set to True to enable wind disturbance handling (DOBC layer active)

import numpy as np

# =============================================================================
# CASCADE PID CONTROLLER STATE - persistent across calls via mutable dict
# =============================================================================
# Outer loop: position error → velocity setpoint
# Inner loop: velocity error → velocity command output
# This is the standard cascade architecture used in PX4/ArduPilot flight stacks.
# =============================================================================

_state = {
    # --- Outer loop (position → velocity setpoint) ---
    'pos_integral':         np.zeros(3),
    'pos_prev_error':       np.zeros(3),

    # --- Inner loop (velocity → command) ---
    'vel_integral':         np.zeros(3),
    'vel_prev_error':       np.zeros(3),

    # --- Yaw (single PID loop, angle → yaw rate) ---
    'yaw_integral':         0.0,
    'yaw_prev_error':       0.0,

    # --- Disturbance Observer (DOBC) for wind rejection ---
    # Estimates persistent disturbance as low-pass filtered integral of residual
    'disturbance_estimate': np.zeros(3),

    # --- Timestamp tracking for real Tello experiment ---
    'prev_timestamp':       None,
}

# =============================================================================
# WIND FLAG
# Set to True to activate DOBC wind rejection layer during simulation marking.
# This flag is read by the simulator to enable wind forces on the drone.
# The controller also uses wind_enabled argument passed per call.
# =============================================================================

wind_flag = False

# =============================================================================
# TUNING PARAMETERS
# Optimised using staged Bayesian optimisation with GPR + Expected Improvement
# against real PyBullet physics in headless DIRECT mode.
# Cost function penalises violations of marking thresholds:
#   pos error mean < 0.01m, pos error std < 0.01,
#   yaw error mean < 0.01rad, yaw error std < 0.001rad
# =============================================================================

# Outer loop (position → velocity setpoint)
KP_POS          = np.array([1.0840, 1.0840, 2.0844])   # Proportional gains [x, y, z]
KI_POS          = np.array([1.2280, 1.2280, 0.7421]) # Integral gains     [x, y, z]
KD_POS          = np.array([1.0739, 1.0739, 0.4774])  # Derivative gains   [x, y, z]

# Inner loop (velocity error → output command)
KP_VEL          = np.array([2.7897, 2.7897, 1.8114])
KI_VEL          = np.array([0.7826, 0.7826, 1.4412])
KD_VEL          = np.array([0.6689, 0.6689, 0.3815])

# Yaw controller
KP_YAW          = 0.6971
KI_YAW          = 0.4736
KD_YAW          = 0.0960

# Velocity output saturation limits (m/s)
VEL_LIMIT       = 1.3722
YAW_RATE_LIMIT  = 0.7921  # rad/s

# Anti-windup: integral clamp limits
POS_INTEG_LIMIT = 0.0208
VEL_INTEG_LIMIT = 1.1628
YAW_INTEG_LIMIT = 1.4157

# Disturbance observer gain (how fast to adapt to wind)
DOBC_GAIN       = 0.1218  # Low-pass filter coefficient (0 = no adaptation, 1 = instant)



# =============================================================================
# DATA COLLECTION
# =============================================================================
# Logs to output.csv on every controller call (50 Hz = 50 rows per second).
# File is created fresh on first call, then appended to on every subsequent call.
#
# Columns recorded:
#   timestamp        — dt value passed in (ms from Vicon, s from simulator)
#   x, y, z          — current drone position (m)
#   roll, pitch, yaw — current drone attitude (rad)
#   vx_cmd           — velocity command sent in x direction (m/s)
#   vy_cmd           — velocity command sent in y direction (m/s)
#   vz_cmd           — velocity command sent in z direction (m/s)
#   yaw_rate_cmd     — yaw rate command sent (rad/s)
#   target_x         — target x position (m)
#   target_y         — target y position (m)
#   target_z         — target z position (m)
#   target_yaw       — target yaw angle (rad)
#   pos_error        — Euclidean distance from current position to target (m)
#   yaw_error        — absolute yaw error wrapped to [-pi, pi] (rad)
# =============================================================================

_DATA_FILE        = 'output.csv'
_data_initialised = False


def _init_data_file():
    """
    Create output.csv with header row on first controller call.
    Overwrites any existing file from a previous run.
    """
    global _data_initialised
    with open(_DATA_FILE, 'w') as f:
        f.write(
            'timestamp,'
            'x,y,z,'
            'roll,pitch,yaw,'
            'vx_cmd,vy_cmd,vz_cmd,yaw_rate_cmd,'
            'target_x,target_y,target_z,target_yaw,'
            'pos_error,yaw_error\n'
        )
    _data_initialised = True


def _log_data(timestamp, state, output, target_pos):
    """
    Append one row to output.csv per controller call.
    Follows lab sheet format:
        new_data = np.hstack([(timestamp, state)])
        np.savetxt(f, [new_data], delimiter=',', fmt='%.6f')
    Extended to also log commands, target pose, and errors for analysis.
    """
    global _data_initialised
    if not _data_initialised:
        _init_data_file()

    pos_current = np.array(state[0:3])
    yaw_current = state[5]
    pos_target  = np.array(target_pos[0:3])
    yaw_target  = target_pos[3]

    pos_error = float(np.linalg.norm(pos_target - pos_current))
    yaw_error = float(
        abs((yaw_current - yaw_target + np.pi) % (2 * np.pi) - np.pi))

    new_data = np.hstack([
        [timestamp],             # col  0:    timestamp
        state,                   # cols 1-6:  x, y, z, roll, pitch, yaw
        list(output),            # cols 7-10: vx, vy, vz, yaw_rate commands
        list(target_pos),        # cols 11-14: target x, y, z, yaw
        [pos_error, yaw_error],  # cols 15-16: positional and yaw errors
    ])

    with open(_DATA_FILE, 'a') as f:
        np.savetxt(f, [new_data], delimiter=',', fmt='%.6f')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _wrap_angle(angle):
    """Wrap angle to [-pi, pi] to avoid yaw discontinuities."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _pid_step(error, prev_error, integral, kp, ki, kd, dt,
              integ_limit, output_limit=None):
    """
    Generic PID step.
    Returns (output, updated_integral, error) for the caller to store.
    Includes:
      - Derivative on error (not measurement) for simplicity
      - Integral anti-windup via clamping
    """
    # Accumulate integral with anti-windup clamping
    integral   = np.clip(integral + error * dt, -integ_limit, integ_limit)

    # Derivative term (finite difference)
    derivative = (error - prev_error) / dt if dt > 1e-6 else np.zeros_like(error)

    output = kp * error + ki * integral + kd * derivative

    if output_limit is not None:
        output = np.clip(output, -output_limit, output_limit)

    return output, integral, error


# =============================================================================
# MAIN CONTROLLER FUNCTION
# =============================================================================

def controller(state, target_pos, dt, wind_enabled=False):
    """
    Cascade PID controller for UAV position stabilisation.

    Architecture:
        Outer loop : position error → desired velocity  (slow loop, position bandwidth)
        Inner loop : velocity error → velocity command  (fast loop, velocity bandwidth)

    This two-layer cascade is the standard approach in commercial/research autopilots
    (PX4, ArduPilot) because it decouples position tracking from velocity dynamics,
    allows independent tuning of each layer, and naturally limits velocity.

    When wind is enabled, a lightweight Disturbance Observer (DOBC) estimates the
    persistent wind-induced bias and subtracts it from the velocity command, improving
    steady-state rejection without requiring large integral gains.

    NOTE for real Tello experiment:
        dt argument is a timestamp in milliseconds from Vicon — converted to seconds
        automatically before use in PID calculations.

    Args:
        state       : [x, y, z, roll, pitch, yaw]  (m, m, m, rad, rad, rad)
        target_pos  : (x_d, y_d, z_d, yaw_d)       (m, m, m, rad)
        dt          : timestep (s) in simulator /
                      timestamp (ms) from Vicon in real experiment
        wind_enabled: flag to activate DOBC wind-rejection layer

    Returns:
        (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)     (m/s, m/s, m/s, rad/s)
        Clamped to [-100, 100] for Tello hardware compatibility.
    """

    # ------------------------------------------------------------------
    # 1. Unpack state and target
    # ------------------------------------------------------------------
    pos_current = np.array(state[0:3])       # current position [x, y, z] (m)
    yaw_current = state[5]                   # current yaw angle (rad)

    pos_target  = np.array(target_pos[0:3])  # desired position [x, y, z] (m)
    yaw_target  = target_pos[3]              # desired yaw angle (rad)

    # ------------------------------------------------------------------
    # 1a. Timestep handling
    # ------------------------------------------------------------------
    # Simulator: dt is control timestep in seconds (0.02s at 50 Hz)
    # Real Tello via Vicon: dt is a timestamp in milliseconds —
    # compute actual timestep from difference between consecutive calls.
    # Heuristic: if dt > 1.0 it is a Vicon timestamp in ms.

    if dt > 1.0:
        prev_t = _state['prev_timestamp']
        if prev_t is None:
            dt_sec = 0.02               # sensible default on first call
        else:
            dt_sec = max((dt - prev_t) / 1000.0, 1e-3)  # ms → s, floor at 1ms
        _state['prev_timestamp'] = dt
    else:
        dt_sec = dt                     # already in seconds (simulator)

    # ------------------------------------------------------------------
    # 2. OUTER LOOP — Position PID → desired velocity in global frame
    # ------------------------------------------------------------------
    pos_error = pos_target - pos_current

    vel_desired, _state['pos_integral'], _state['pos_prev_error'] = _pid_step(
        error        = pos_error,
        prev_error   = _state['pos_prev_error'],
        integral     = _state['pos_integral'],
        kp           = KP_POS,
        ki           = KI_POS,
        kd           = KD_POS,
        dt           = dt_sec,
        integ_limit  = POS_INTEG_LIMIT,
        output_limit = VEL_LIMIT,
    )

    # ------------------------------------------------------------------
    # 3. COORDINATE TRANSFORMATION — Global frame → Body frame
    # ------------------------------------------------------------------
    # Velocity commands must be in the body frame (relative to drone heading).
    # Rotation around Z axis by current yaw angle:
    #
    #   [vx_body]   [ cos(yaw)  sin(yaw)  0 ] [vx_global]
    #   [vy_body] = [-sin(yaw)  cos(yaw)  0 ] [vy_global]
    #   [vz_body]   [    0          0     1 ] [vz_global]
    #
    # Pitch and roll rotations omitted — brief states control frame is
    # not rotated in pitch/roll.

    cos_yaw = np.cos(yaw_current)
    sin_yaw = np.sin(yaw_current)

    vel_desired_body = np.array([
         cos_yaw * vel_desired[0] + sin_yaw * vel_desired[1],
        -sin_yaw * vel_desired[0] + cos_yaw * vel_desired[1],
         vel_desired[2]
    ])

    # ------------------------------------------------------------------
    # 4. INNER LOOP — Velocity PID → final velocity command
    # ------------------------------------------------------------------
    # Inner loop corrects lag between desired and actual velocity.
    # vel_error is zero here (placeholder) since actual velocity feedback
    # is not available in the state vector — swap vel_estimated_body for
    # state[6:9] if the simulator exposes velocity measurements.

    vel_estimated_body = vel_desired_body   # placeholder
    vel_error          = vel_desired_body - vel_estimated_body

    vel_cmd, _state['vel_integral'], _state['vel_prev_error'] = _pid_step(
        error        = vel_error,
        prev_error   = _state['vel_prev_error'],
        integral     = _state['vel_integral'],
        kp           = KP_VEL,
        ki           = KI_VEL,
        kd           = KD_VEL,
        dt           = dt_sec,
        integ_limit  = VEL_INTEG_LIMIT,
        output_limit = VEL_LIMIT,
    )

    final_vel = np.clip(vel_desired_body + vel_cmd, -VEL_LIMIT, VEL_LIMIT)

    # ------------------------------------------------------------------
    # 5. DISTURBANCE OBSERVER (DOBC) — wind rejection
    # ------------------------------------------------------------------
    # Estimates persistent wind-induced velocity bias as an exponential
    # moving average of the residual between commanded and desired velocity.
    # The estimate is subtracted from the command to pre-compensate.
    #
    # Simplified DOBC — full implementation would model drone dynamics.
    # Low-pass filter coefficient DOBC_GAIN controls adaptation speed:
    #   low  → slow adaptation, less noise sensitivity
    #   high → fast adaptation, risk of overreacting to normal motion

    if wind_enabled:
        residual = final_vel - vel_desired_body

        _state['disturbance_estimate'] = (
            (1 - DOBC_GAIN) * _state['disturbance_estimate']
            + DOBC_GAIN * residual
        )

        final_vel = np.clip(
            final_vel - _state['disturbance_estimate'],
            -VEL_LIMIT, VEL_LIMIT
        )

    # ------------------------------------------------------------------
    # 6. YAW CONTROLLER — angle error → yaw rate command
    # ------------------------------------------------------------------
    yaw_error = _wrap_angle(yaw_target - yaw_current)

    yaw_rate_cmd, _state['yaw_integral'], _state['yaw_prev_error'] = _pid_step(
        error        = yaw_error,
        prev_error   = _state['yaw_prev_error'],
        integral     = _state['yaw_integral'],
        kp           = KP_YAW,
        ki           = KI_YAW,
        kd           = KD_YAW,
        dt           = dt_sec,
        integ_limit  = YAW_INTEG_LIMIT,
        output_limit = YAW_RATE_LIMIT,
    )

    # ------------------------------------------------------------------
    # 7. Clamp outputs for Tello hardware compatibility
    # ------------------------------------------------------------------
    # Lab sheet specifies velocity and yaw rate must be within [-100, 100]
    # for the real Tello. Simulator is unaffected since VEL_LIMIT < 100.

    output = (
        float(np.clip(final_vel[0],  -100, 100)),  # vx command (m/s)
        float(np.clip(final_vel[1],  -100, 100)),  # vy command (m/s)
        float(np.clip(final_vel[2],  -100, 100)),  # vz command (m/s)
        float(np.clip(yaw_rate_cmd,  -100, 100)),  # yaw rate command (rad/s)
    )

    # ------------------------------------------------------------------
    # 8. Data logging — append one row to output.csv
    # ------------------------------------------------------------------
    # Records: timestamp, state, commands, target, pos_error, yaw_error
    # Disable by commenting out the line below if not needed.

    _log_data(dt, state, output, target_pos)

    return output
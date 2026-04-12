%% tello_position_loop_model.m
% Closer-to-run.py MATLAB model:
% - outer loop: current controller.py logic at 50 Hz
% - inner loop: src/tello_controller.py cascade at 1000 Hz
% - actuator: first-order motor RPM model
% - rigid body: full 6DOF translation + rotation using quaternion state
%
% This still is not identical to PyBullet, but it is much closer than the
% previous first-order outer-loop-only approximation.

clear; clc; close all;

%% ================= USER SETTINGS =================
T_end = 21.0;             % 1 s pre-step + 10 s settle + 10 s statistics
dt_inner = 1/1000;        % matches run.py physics loop
dt_outer = 1/50;          % matches run.py outer loop
outer_decim = round(dt_outer / dt_inner);

% Initial condition
pos0 = [0; 0; 1.0];
eul0 = [0; 0; 0];         % [roll pitch yaw]
quat0 = quat_from_euler_zyx(eul0);
vel_world0 = [0; 0; 0];
omega_body0 = [0; 0; 0];
rpm0 = [0; 0; 0; 0];

% Reference before and after the step
ref0 = [0, 0, 1.0, 0];
ref1 = load_first_target_or_default([2, 2, 2, 0]);
t_step = 1.0;

% Wind option: run.py defaults to disabled
wind_enabled = false;
wind_force_world = [0.0; 0.0; 0.0];    % constant world-frame disturbance force [N]

%% ================= PARAMETERS FROM PYTHON =================
P.g = 9.81;
P.M = 0.088;
P.L = 0.06;
P.I_body = diag([0.00679, 0.00679, 0.01313]);  % from tello.urdf
P.KF = 0.566e-5;
P.KM = 0.762e-7;
P.K_TRANS = [3.365e-2; 3.365e-2; 3.365e-2];
P.TM = 0.0163;
P.max_angle = 0.35;
P.max_rpm = 28000.0;
P.max_motor_command = 0.75202525252;
P.max_yaw_rate = 1.74533;
P.max_vel_ctrl = [1.08; 1.08; 1.08];         % controller.py internal clip
P.max_vel_checked = [1.0; 1.0; 1.0];         % run.py check_action() clip

P.outer.kp_pos = [0.53; 0.53; 1.18];
P.outer.ki_pos = [0.36; 0.36; 0.60];
P.outer.kd_vel = [0.16; 0.16; 0.96];
P.outer.ki_pos_sat = [0.48; 0.48; 0.58];
P.outer.kp_yaw = 2.05;
P.outer.ki_yaw = 0.42;
P.outer.kd_yaw = 0.16;
P.outer.ki_yaw_sat = 0.18;
P.outer.k_dob = [0.35; 0.35; 0.25];
P.outer.dob_leak = [0.32; 0.32; 0.24];
P.outer.k_comp = [0.60; 0.60; 0.50];
P.outer.int_pos_leak = [2.5; 2.5; 1.5];
P.outer.int_yaw_leak = 2.0;
P.outer.vel_est_lpf_alpha = 0.56;
P.outer.yaw_align_tol = 0.167;

P.vel_pid = make_pid(7.0, 0.6, 0.2, [10; 10; 10]);
P.att_pid = make_pid(0.2, 0.01, 0.015, [1; 1; 1]);
P.rate_pid = make_pid(0.05, 0.2, 0.0, [0.1; 0.1; 0.1]);

P.mixing_matrix = [
    0.25, -1/(4*P.L), -1/(4*P.L), -P.KF/(4*P.KM);
    0.25,  1/(4*P.L),  1/(4*P.L), -P.KF/(4*P.KM);
    0.25,  1/(4*P.L), -1/(4*P.L),  P.KF/(4*P.KM);
    0.25, -1/(4*P.L),  1/(4*P.L),  P.KF/(4*P.KM)
];

%% ================= SIMULATION BUFFERS =================
N = floor(T_end / dt_inner) + 1;
t = (0:N-1)' * dt_inner;

pos_log = zeros(N, 3);
quat_log = zeros(N, 4);          % [w x y z]
eul_log = zeros(N, 3);
vel_world_log = zeros(N, 3);
vel_yaw_body_log = zeros(N, 3);
omega_body_log = zeros(N, 3);
rpm_log = zeros(N, 4);
ref_log = zeros(N, 4);
desired_vel_log = zeros(N, 3);
yaw_rate_cmd_log = zeros(N, 1);
force_body_log = zeros(N, 3);
torque_body_log = zeros(N, 3);
pos_err_world_log = zeros(N, 3);
pos_err_body_log = zeros(N, 3);
vel_est_world_log = zeros(N, 3);

pos_log(1,:) = pos0.';
quat_log(1,:) = quat0.';
eul_log(1,:) = eul0.';
vel_world_log(1,:) = vel_world0.';
omega_body_log(1,:) = omega_body0.';
rpm_log(1,:) = rpm0.';

outer_mem.int_pos_body = zeros(3,1);
outer_mem.int_yaw = 0.0;
outer_mem.prev_pos_error_world = [];
outer_mem.prev_yaw_err = [];
outer_mem.vel_est_world = zeros(3,1);
outer_mem.d_hat = zeros(3,1);

vel_pid_state = P.vel_pid;
att_pid_state = P.att_pid;
rate_pid_state = P.rate_pid;

desired_vel_hold = zeros(3,1);
yaw_rate_hold = 0.0;
outer_counter = 0;

metric_t = [];
metric_pos_err = [];
metric_yaw_err = [];
window_start = t_step + 10.0;
window_end = t_step + 20.0;

%% ================= MAIN LOOP =================
for k = 1:N-1
    tk = t(k);

    pos = pos_log(k,:).';
    quat = quat_log(k,:).';
    quat = quat_normalize(quat);
    vel_world = vel_world_log(k,:).';
    omega_body = omega_body_log(k,:).';
    rpm_prev = rpm_log(k,:).';

    eul = quat_to_euler_zyx(quat);
    yaw = eul(3);

    if tk < t_step
        ref = ref0(:);
    else
        ref = ref1(:);
    end
    ref_log(k,:) = ref.';

    pos_err_world = ref(1:3) - pos;
    pos_err_world_log(k,:) = pos_err_world.';

    R_w2b_yaw = rot_world_to_yaw_body(yaw);
    vel_yaw_body = R_w2b_yaw * vel_world;
    vel_yaw_body_log(k,:) = vel_yaw_body.';

    outer_counter = outer_counter + 1;
    if outer_counter >= outer_decim
        outer_counter = 0;
        state6 = [pos; eul];
        [desired_vel_hold, yaw_rate_hold, outer_mem, dbg] = outer_controller_step( ...
            state6, ref, dt_outer, wind_enabled, outer_mem, P);
        pos_err_body_log(k,:) = dbg.pos_err_body.';
        vel_est_world_log(k,:) = dbg.vel_est_world.';

        if tk >= window_start && tk < window_end
            metric_t(end+1,1) = tk; %#ok<SAGROW>
            metric_pos_err(end+1,1) = norm(pos - ref(1:3)); %#ok<SAGROW>
            metric_yaw_err(end+1,1) = wrap_to_pi(yaw - ref(4)); %#ok<SAGROW>
        end
    else
        pos_err_body_log(k,:) = pos_err_body_log(max(k-1,1),:);
        vel_est_world_log(k,:) = vel_est_world_log(max(k-1,1),:);
    end

    desired_vel_log(k,:) = desired_vel_hold.';
    yaw_rate_cmd_log(k) = yaw_rate_hold;

    [rpm_desired, vel_pid_state, att_pid_state, rate_pid_state] = inner_control_step( ...
        desired_vel_hold, vel_yaw_body, quat, omega_body, yaw_rate_hold, ...
        dt_inner, vel_pid_state, att_pid_state, rate_pid_state, P);

    rpm = rpm_prev + (rpm_desired - rpm_prev) / P.TM * dt_inner;
    rpm = min(max(rpm, 0.0), P.max_rpm);

    [force_body, torque_body] = compute_body_force_torque(rpm, vel_world, quat, P);
    force_body_log(k,:) = force_body.';
    torque_body_log(k,:) = torque_body.';

    R_b2w = quat_to_rotm(quat);
    accel_world = (R_b2w * force_body + wind_force_world) / P.M + [0; 0; -P.g];
    omega_dot = P.I_body \ (torque_body - cross(omega_body, P.I_body * omega_body));

    vel_world_next = vel_world + accel_world * dt_inner;
    pos_next = pos + vel_world * dt_inner;
    omega_body_next = omega_body + omega_dot * dt_inner;
    quat_dot = quat_derivative_body(quat, omega_body);
    quat_next = quat_normalize(quat + quat_dot * dt_inner);

    pos_log(k+1,:) = pos_next.';
    quat_log(k+1,:) = quat_next.';
    eul_log(k+1,:) = quat_to_euler_zyx(quat_next).';
    vel_world_log(k+1,:) = vel_world_next.';
    omega_body_log(k+1,:) = omega_body_next.';
    rpm_log(k+1,:) = rpm.';
end

ref_log(end,:) = ref_log(end-1,:);
desired_vel_log(end,:) = desired_vel_log(end-1,:);
yaw_rate_cmd_log(end) = yaw_rate_cmd_log(end-1);
force_body_log(end,:) = force_body_log(end-1,:);
torque_body_log(end,:) = torque_body_log(end-1,:);
pos_err_world_log(end,:) = pos_err_world_log(end-1,:);
pos_err_body_log(end,:) = pos_err_body_log(end-1,:);
vel_est_world_log(end,:) = vel_est_world_log(end-1,:);
vel_yaw_body_log(end,:) = vel_yaw_body_log(end-1,:);

%% ================= METRICS =================
if isempty(metric_pos_err)
    warning('No metric samples collected in the requested window.');
    mean_err = NaN; std_err = NaN; mean_yaw_abs = NaN; std_yaw = NaN;
else
    mean_err = mean(metric_pos_err);
    std_err = std(metric_pos_err);
    mean_yaw_abs = mean(abs(metric_yaw_err));
    std_yaw = std(metric_yaw_err);
end

fprintf('\n========== MATLAB rigid-body model (closer to run.py) ==========' );
fprintf('\nStatistics window: [%.1f s, %.1f s)', window_start, window_end);
fprintf('\nMean position error norm: %.6f m', mean_err);
fprintf('\nStd  position error norm: %.6f m', std_err);
fprintf('\nMean |yaw error|       : %.6f rad', mean_yaw_abs);
fprintf('\nStd  yaw error         : %.6f rad', std_yaw);
fprintf('\nMetric samples         : %d (outer-loop 50 Hz samples)\n', numel(metric_pos_err));

%% ================= PLOTS =================
figure('Color','w','Position',[80 50 1200 860]);
tiledlayout(5,1,'Padding','compact','TileSpacing','compact');

nexttile;
plot(t, pos_log(:,1), 'LineWidth', 1.1); hold on;
plot(t, pos_log(:,2), 'LineWidth', 1.1);
plot(t, pos_log(:,3), 'LineWidth', 1.1);
plot(t, ref_log(:,1), '--');
plot(t, ref_log(:,2), '--');
plot(t, ref_log(:,3), '--');
xline(t_step, ':k');
grid on; xlim([0 T_end]);
legend('x','y','z','x_{ref}','y_{ref}','z_{ref}','Location','best');
title('Position (world)'); ylabel('m');

nexttile;
plot(t, vel_yaw_body_log(:,1), 'LineWidth', 1.1); hold on;
plot(t, vel_yaw_body_log(:,2), 'LineWidth', 1.1);
plot(t, vel_yaw_body_log(:,3), 'LineWidth', 1.1);
plot(t, desired_vel_log(:,1), '--');
plot(t, desired_vel_log(:,2), '--');
plot(t, desired_vel_log(:,3), '--');
xline(t_step, ':k');
grid on; xlim([0 T_end]);
legend('v_x^{yaw}','v_y^{yaw}','v_z^{yaw}','v_{cmd,x}','v_{cmd,y}','v_{cmd,z}','Location','best');
title('Velocity tracked by inner loop'); ylabel('m/s');

nexttile;
plot(t, rad2deg(eul_log(:,1)), 'LineWidth', 1.1); hold on;
plot(t, rad2deg(eul_log(:,2)), 'LineWidth', 1.1);
plot(t, rad2deg(eul_log(:,3)), 'LineWidth', 1.1);
plot(t, rad2deg(ref_log(:,4)), '--k', 'LineWidth', 1.0);
xline(t_step, ':k');
grid on; xlim([0 T_end]);
legend('\phi','\theta','\psi','\psi_{ref}','Location','best');
title('Euler angles'); ylabel('deg');

nexttile;
plot(t, rpm_log(:,1), 'LineWidth', 1.1); hold on;
plot(t, rpm_log(:,2), 'LineWidth', 1.1);
plot(t, rpm_log(:,3), 'LineWidth', 1.1);
plot(t, rpm_log(:,4), 'LineWidth', 1.1);
xline(t_step, ':k');
grid on; xlim([0 T_end]);
legend('rpm_1','rpm_2','rpm_3','rpm_4','Location','best');
title('Motor RPM'); ylabel('rpm');

nexttile;
plot(t, sqrt(sum(pos_err_world_log.^2,2)), 'k', 'LineWidth', 1.4); hold on;
plot(t, arrayfun(@(a,b) wrap_to_pi(a-b), eul_log(:,3), ref_log(:,4)), 'LineWidth', 1.0);
xline(t_step, ':k');
xline(window_start, '--g');
xline(window_end, '--g');
grid on; xlim([0 T_end]);
legend('||e_p||','e_{\psi}','Location','best');
title('Tracking errors'); xlabel('Time [s]'); ylabel('m / rad');

sgtitle(sprintf(['Rigid-body MATLAB approximation of run.py | dt_{inner}=%.4f s | ', ...
    'dt_{outer}=%.3f s | wind=%d'], dt_inner, dt_outer, wind_enabled));

%% ================= LOCAL FUNCTIONS =================
function pid = make_pid(Kp, Ki, Kd, Ki_sat)
    pid.Kp = Kp;
    pid.Ki = Ki;
    pid.Kd = Kd;
    pid.Ki_sat = Ki_sat(:);
    pid.int = zeros(3,1);
    pid.previous_error = zeros(3,1);
end

function ref = load_first_target_or_default(default_ref)
    candidate = fullfile(fileparts(mfilename('fullpath')), 'assignment_3_2026', 'targets.csv');
    if exist(candidate, 'file') ~= 2
        ref = default_ref(:).';
        return;
    end

    try
        T = readmatrix(candidate);
        if size(T,1) >= 1 && size(T,2) >= 4 && all(isfinite(T(1,1:4)))
            ref = T(1,1:4);
        else
            ref = default_ref(:).';
        end
    catch
        ref = default_ref(:).';
    end
end

function [output, pid] = pid_control_update(pid, error, timestep)
    pid.int = pid.int + error * timestep;
    pid.int = min(max(pid.int, -pid.Ki_sat), pid.Ki_sat);
    derivative = (error - pid.previous_error) / timestep;
    pid.previous_error = error;
    output = pid.Kp * error + pid.Ki * pid.int + pid.Kd * derivative;
end

function [vel_cmd, yaw_rate_cmd, mem, dbg] = outer_controller_step(state6, target_pos, dt, wind_enabled, mem, P)
    x = state6(1); y = state6(2); z = state6(3);
    yaw = state6(6);

    pos = [x; y; z];
    pos_d = target_pos(1:3);
    yaw_d = target_pos(4);

    pos_err_world = pos_d - pos;
    if ~isempty(mem.prev_pos_error_world) && dt > 1e-6
        pos_derivative = (pos_err_world - mem.prev_pos_error_world) / dt;
        pos_derivative = min(max(pos_derivative, -3.0), 3.0);
        raw_vel = -pos_derivative;
        a = P.outer.vel_est_lpf_alpha;
        mem.vel_est_world = a * raw_vel + (1.0 - a) * mem.vel_est_world;
    end
    mem.prev_pos_error_world = pos_err_world;

    R_w2b_yaw = rot_world_to_yaw_body(yaw);
    pos_err_body = R_w2b_yaw * pos_err_world;
    vel_est_body = R_w2b_yaw * mem.vel_est_world;

    yaw_err = wrap_to_pi(yaw_d - yaw);
    yaw_aligned = abs(yaw_err) < P.outer.yaw_align_tol;

    mem.int_pos_body(3) = mem.int_pos_body(3) + pos_err_body(3) * dt;
    if yaw_aligned
        mem.int_pos_body(1:2) = mem.int_pos_body(1:2) + pos_err_body(1:2) * dt;
    end
    mem.int_pos_body = mem.int_pos_body .* exp(-P.outer.int_pos_leak * dt);
    mem.int_pos_body = min(max(mem.int_pos_body, -P.outer.ki_pos_sat), P.outer.ki_pos_sat);

    p_term = P.outer.kp_pos .* pos_err_body;
    i_term = P.outer.ki_pos .* mem.int_pos_body;
    d_term = -P.outer.kd_vel .* vel_est_body;
    vel_cmd_nominal = p_term + i_term + d_term;

    if wind_enabled
        if norm(pos_err_body(1:2)) > 0.8
            vel_tracking_error = vel_cmd_nominal - vel_est_body;
            d_dot = P.outer.k_dob .* vel_tracking_error - P.outer.dob_leak .* mem.d_hat;
            mem.d_hat = mem.d_hat + d_dot * dt;
            mem.d_hat = min(max(mem.d_hat, -0.55), 0.55);
        end
        vel_cmd = vel_cmd_nominal + P.outer.k_comp .* mem.d_hat;
    else
        mem.d_hat(:) = 0.0;
        vel_cmd = vel_cmd_nominal;
    end

    vel_cmd = min(max(vel_cmd, -P.max_vel_ctrl), P.max_vel_ctrl);
    if ~yaw_aligned
        vel_cmd(1:2) = 0.0;
    end

    mem.int_yaw = mem.int_yaw + yaw_err * dt;
    mem.int_yaw = mem.int_yaw * exp(-P.outer.int_yaw_leak * dt);
    mem.int_yaw = min(max(mem.int_yaw, -P.outer.ki_yaw_sat), P.outer.ki_yaw_sat);

    if isempty(mem.prev_yaw_err) || dt <= 1e-6
        yaw_err_deriv = 0.0;
    else
        delta = wrap_to_pi(yaw_err - mem.prev_yaw_err);
        yaw_err_deriv = min(max(delta / dt, -10.0), 10.0);
    end
    mem.prev_yaw_err = yaw_err;

    yaw_rate_unsat = P.outer.kp_yaw * yaw_err + ...
                     P.outer.ki_yaw * mem.int_yaw + ...
                     P.outer.kd_yaw * yaw_err_deriv;
    yaw_rate_cmd = min(max(yaw_rate_unsat, -P.max_yaw_rate), P.max_yaw_rate);
    if abs(yaw_rate_cmd) >= P.max_yaw_rate - 1e-6
        mem.int_yaw = mem.int_yaw - yaw_err * dt;
    end

    vel_cmd = min(max(vel_cmd, -P.max_vel_checked), P.max_vel_checked);
    yaw_rate_cmd = min(max(yaw_rate_cmd, -P.max_yaw_rate), P.max_yaw_rate);

    dbg.pos_err_world = pos_err_world;
    dbg.pos_err_body = pos_err_body;
    dbg.vel_est_world = mem.vel_est_world;
    dbg.vel_est_body = vel_est_body;
    dbg.vel_cmd_nominal = vel_cmd_nominal;
end

function [rpm_desired, vel_pid_state, att_pid_state, rate_pid_state] = inner_control_step( ...
    desired_vel, lin_vel_yaw_body, quat, omega_body, yaw_rate_setpoint, ...
    timestep, vel_pid_state, att_pid_state, rate_pid_state, P)

    vel_error = desired_vel - lin_vel_yaw_body;
    [desired_accel, vel_pid_state] = pid_control_update(vel_pid_state, vel_error, timestep);

    desired_accel(3) = desired_accel(3) + P.g;
    thrust = desired_accel(3) * P.M;

    current_angle = quat_to_euler_zyx(quat);
    if norm(desired_accel(1:2)) < 1e-6
        desired_angle = [0; 0; 0];
    else
        desired_angle = [
            atan2(-desired_accel(2), desired_accel(3));
            atan2( desired_accel(1), desired_accel(3));
            0
        ];
    end
    desired_angle(1) = min(max(desired_angle(1), -P.max_angle), P.max_angle);
    desired_angle(2) = min(max(desired_angle(2), -P.max_angle), P.max_angle);
    desired_angle(3) = current_angle(3);

    angle_error = desired_angle - current_angle;
    [desired_rate, att_pid_state] = pid_control_update(att_pid_state, angle_error, timestep);
    desired_rate(3) = yaw_rate_setpoint;

    rate_error = desired_rate - omega_body;
    [desired_torque, rate_pid_state] = pid_control_update(rate_pid_state, rate_error, timestep);

    controls = [thrust; desired_torque];
    motor_commands = P.mixing_matrix * controls;
    motor_commands = min(max(motor_commands, 0.0), P.max_motor_command);

    rpm_desired = sqrt(motor_commands / P.KF) * (60 / (2*pi));
    rpm_desired = min(max(rpm_desired, 0.0), P.max_rpm);
end

function [force_body, torque_body] = compute_body_force_torque(rpm_values, vel_world, quat, P)
    R_b2w = quat_to_rotm(quat);
    vel_body = R_b2w.' * vel_world;

    omega = rpm_values * (2*pi/60);
    omega_sq = omega.^2;
    motor_forces = omega_sq * P.KF;

    thrust = [0; 0; sum(motor_forces)];
    drag_body = -P.K_TRANS .* vel_body;
    force_body = drag_body + thrust;

    z_torques = omega_sq * P.KM;
    z_torque = -z_torques(1) - z_torques(2) + z_torques(3) + z_torques(4);
    x_torque = (-motor_forces(1) + motor_forces(2) + motor_forces(3) - motor_forces(4)) * P.L;
    y_torque = (-motor_forces(1) + motor_forces(2) - motor_forces(3) + motor_forces(4)) * P.L;
    torque_body = [x_torque; y_torque; z_torque];
end

function a = wrap_to_pi(a)
    a = mod(a + pi, 2*pi) - pi;
end

function R = rot_world_to_yaw_body(yaw)
    c = cos(yaw);
    s = sin(yaw);
    R = [c, s, 0; -s, c, 0; 0, 0, 1];
end

function q = quat_from_euler_zyx(eul)
    roll = eul(1); pitch = eul(2); yaw = eul(3);
    cr = cos(roll/2);  sr = sin(roll/2);
    cp = cos(pitch/2); sp = sin(pitch/2);
    cy = cos(yaw/2);   sy = sin(yaw/2);
    q = [
        cy*cp*cr + sy*sp*sr;
        cy*cp*sr - sy*sp*cr;
        cy*sp*cr + sy*cp*sr;
        sy*cp*cr - cy*sp*sr
    ];
    q = quat_normalize(q);
end

function eul = quat_to_euler_zyx(q)
    q = quat_normalize(q);
    w = q(1); x = q(2); y = q(3); z = q(4);

    sinr_cosp = 2 * (w*x + y*z);
    cosr_cosp = 1 - 2 * (x*x + y*y);
    roll = atan2(sinr_cosp, cosr_cosp);

    sinp = 2 * (w*y - z*x);
    sinp = min(max(sinp, -1.0), 1.0);
    pitch = asin(sinp);

    siny_cosp = 2 * (w*z + x*y);
    cosy_cosp = 1 - 2 * (y*y + z*z);
    yaw = atan2(siny_cosp, cosy_cosp);

    eul = [roll; pitch; yaw];
end

function R = quat_to_rotm(q)
    q = quat_normalize(q);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y);
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x);
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)
    ];
end

function qdot = quat_derivative_body(q, omega_body)
    omega_quat = [0; omega_body(:)];
    qdot = 0.5 * quat_multiply(q, omega_quat);
end

function q = quat_multiply(q1, q2)
    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);
    q = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2;
        w1*x2 + x1*w2 + y1*z2 - z1*y2;
        w1*y2 - x1*z2 + y1*w2 + z1*x2;
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ];
end

function q = quat_normalize(q)
    nq = norm(q);
    if nq < 1e-12
        q = [1; 0; 0; 0];
    else
        q = q / nq;
    end
end

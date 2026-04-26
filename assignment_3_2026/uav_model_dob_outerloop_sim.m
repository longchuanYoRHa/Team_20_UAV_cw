%% uav_model_dob_outerloop_sim.m
% Model-aware MATLAB simulation for current UAV outer-loop controller.
% Architecture: reference -> outer-loop PID + model-based DOB -> identified
% inner-loop transfer functions -> position/yaw response.
%
% Run this file directly in MATLAB.

clear; clc; close all;
P = defaultParams();

Tend = 30.0;
r0 = [0; 0; 1.0; 0];
r1 = [0; -2.0; 1.0; 0];
tStep = 2.0;
windOn = false;

[t, X, U, DBG] = simulateOuterLoop(P, r0, r1, tStep, Tend, windOn);

idx = (t >= tStep + 10.0) & (t <= tStep + 20.0);
posErr = X(:,1:3) - repmat(r1(1:3).', length(t), 1);
posErrNorm = vecnorm(posErr, 2, 2);
yawErr = wrapToPiVec(X(:,7) - r1(4));

fprintf('\n========== Coursework-style window: %.1f to %.1f s =========\n', tStep+10, tStep+20);
fprintf('Position error norm mean: %.5f m\n', mean(posErrNorm(idx)));
fprintf('Position error norm std : %.5f m\n', std(posErrNorm(idx)));
fprintf('Yaw abs error mean      : %.5f rad\n', mean(abs(yawErr(idx))));
fprintf('Yaw error std           : %.5f rad\n', std(yawErr(idx)));

figure('Color','w','Position',[100 80 980 780]);
tiledlayout(5,1,'Padding','compact','TileSpacing','compact');

nexttile;
plot(t, X(:,1),'LineWidth',1.2); hold on; plot(t, X(:,2),'LineWidth',1.2); plot(t, X(:,3),'LineWidth',1.2);
yline(r1(1),'--'); yline(r1(2),'--'); yline(r1(3),'--'); grid on;
ylabel('position / m'); legend('x','y','z','x_d','y_d','z_d','Location','best'); title('Position response');

nexttile;
plot(t, X(:,4),'LineWidth',1.2); hold on; plot(t, X(:,5),'LineWidth',1.2); plot(t, X(:,6),'LineWidth',1.2);
plot(t, U(:,1),'--'); plot(t, U(:,2),'--'); plot(t, U(:,3),'--'); grid on;
ylabel('velocity / m/s'); legend('v_x','v_y','v_z','u_x','u_y','u_z','Location','best'); title('Inner-loop output vs outer-loop command');

nexttile;
plot(t, rad2deg(X(:,7)),'LineWidth',1.2); hold on; plot(t, rad2deg(U(:,4)),'--'); yline(rad2deg(r1(4)),'--'); grid on;
ylabel('deg / deg/s'); legend('\psi','\dot\psi_{cmd}','\psi_d','Location','best'); title('Yaw response');

nexttile;
plot(t, DBG(:,1),'LineWidth',1.2); hold on; plot(t, DBG(:,2),'LineWidth',1.2); plot(t, DBG(:,3),'LineWidth',1.2); grid on;
ylabel('DOB / wind'); legend('dhat_x','dhat_y','|wind| approx','Location','best'); title('DOB estimate and wind magnitude');

nexttile;
plot(t, posErrNorm,'LineWidth',1.2); hold on; yline(0.01,'--'); grid on;
ylabel('|e_p| / m'); xlabel('time / s'); title('Position error norm');

try
    s = tf('s');
    Gxy = P.xy.K * P.xy.wn^2 * (P.xy.tau_z*s + 1) / (s^2 + 2*P.xy.zeta*P.xy.wn*s + P.xy.wn^2);
    Gxy = Gxy * padeDelay(P.xy.L,2);
    Gz = P.z.K / (P.z.tau*s + 1) * padeDelay(P.z.L,1);
    figure('Color','w'); bodemag(Gxy, Gz); grid on;
    legend('G_{xy}','G_z','Location','best'); title('Identified inner-loop models');
catch
    fprintf('\nControl System Toolbox not available; skipped Bode plot.\n');
end

%% ============================ Main simulation ============================
function [t, X, U, DBG] = simulateOuterLoop(P, r0, r1, tStep, Tend, windOn)
    dt = P.dt; N = floor(Tend/dt)+1; t = (0:N-1)'*dt;
    X = zeros(N,8); U = zeros(N,4); DBG = zeros(N,3);
    X(1,:) = [r0(1), r0(2), r0(3), 0, 0, 0, r0(4), 0];

    intPos = zeros(3,1); intYaw = 0; prevYawErr = []; prevPosErr = [];
    velEstWorld = zeros(3,1); prevXYCmd = zeros(2,1); dhatXY = zeros(2,1);

    % Inner model states for DOB nominal model
    mxy_x = zeros(2,1); mxy_y = zeros(2,1); qx = zeros(2,1); qy = zeros(2,1);
    hist_mx = zeros(max(3,ceil(P.xy.L/dt)+3),1); hist_my = hist_mx;

    % Actual plant states
    px = zeros(2,1); py = zeros(2,1); pz = 0; pyaw = zeros(2,1);
    hist_px = hist_mx; hist_py = hist_mx;
    hist_pz = zeros(max(3,ceil(P.z.L/dt)+3),1);
    hist_yaw = zeros(max(3,ceil(P.yaw.L/dt)+3),1);

    for k = 1:N-1
        rd = r0; if t(k) >= tStep, rd = r1; end
        pos = X(k,1:3).'; yaw = X(k,7);
        posErrWorld = rd(1:3) - pos;

        if ~isempty(prevPosErr)
            posDeriv = clipVec((posErrWorld-prevPosErr)/dt, -3, 3);
            rawVel = -posDeriv;
            velEstWorld = P.velEstAlpha*rawVel + (1-P.velEstAlpha)*velEstWorld;
        end
        prevPosErr = posErrWorld;

        posErrBody = rotWorldToYawBody(yaw) * posErrWorld;
        velEstBody = rotWorldToYawBody(yaw) * velEstWorld;
        yawErr = wrapToPi(rd(4) - yaw);
        yawAligned = abs(yawErr) < P.yawAlignTol;

        intPos(3) = intPos(3) + posErrBody(3)*dt;
        if yawAligned, intPos(1:2) = intPos(1:2) + posErrBody(1:2)*dt; end
        intPos = clipVec(intPos .* exp(-P.intLeak*dt), -P.kiSat, P.kiSat);

        pidErr = posErrBody;
        pidErr(1) = softDeadzone(pidErr(1), P.softDeadzone);
        pidErr(2) = softDeadzone(pidErr(2), P.softDeadzone);
        pTerm = P.kp .* pidErr; iTerm = P.ki .* intPos; dTerm = -P.kd .* velEstBody;

        xyErr = norm(posErrBody(1:2));
        nearFactor = 1 - smoothstep(xyErr, P.pBoostFullEdge, P.pBoostZeroEdge);
        pTerm(1:2) = (1 + P.pBoostGain*nearFactor) * pTerm(1:2);
        dTerm(1:2) = smoothstep(xyErr, P.dTermZeroEdge, P.dTermFullEdge) * dTerm(1:2);
        velCmdNom = pTerm + iTerm + dTerm;

        [modelVx, mxy_x, hist_mx] = stepSecondOrderZero(velCmdNom(1), mxy_x, hist_mx, P.xy, dt);
        [modelVy, mxy_y, hist_my] = stepSecondOrderZero(velCmdNom(2), mxy_y, hist_my, P.xy, dt);
        mismatch = clipVec(velEstBody(1:2) - [modelVx; modelVy], -1, 1);
        enableScale = double(windOn) + 0.15*double(~windOn);
        [dhatXY(1), qx] = stepSecondOrderLPF(enableScale*mismatch(1), qx, P.qW, P.qZeta, dt);
        [dhatXY(2), qy] = stepSecondOrderLPF(enableScale*mismatch(2), qy, P.qW, P.qZeta, dt);
        dhatXY = clipVec(dhatXY, -P.dhatMax, P.dhatMax);
        dobFade = smoothstep(xyErr, P.dobZeroEdge, P.dobFullEdge);
        velCmd = velCmdNom + [P.kCompXY*dobFade*dhatXY; 0];

        if ~yawAligned, velCmd(1:2)=0; end
        xyCmd = P.cmdXYLpfBeta*prevXYCmd + (1-P.cmdXYLpfBeta)*velCmd(1:2);
        prevXYCmd = xyCmd; velCmd(1:2)=xyCmd;
        velCmd = clipVec(velCmd, -P.maxVel, P.maxVel);
        if norm(velCmd(1:2)) > P.maxXYSpeed, velCmd(1:2) = velCmd(1:2)*P.maxXYSpeed/norm(velCmd(1:2)); end

        intYaw = clipScalar((intYaw + yawErr*dt)*exp(-P.intYawLeak*dt), -P.kiYawSat, P.kiYawSat);
        if isempty(prevYawErr), yawErrD = 0; else, yawErrD = clipScalar(wrapToPi(yawErr-prevYawErr)/dt, -10, 10); end
        prevYawErr = yawErr;
        yawRateCmd = clipScalar(P.kpYaw*yawErr + P.kiYaw*intYaw + P.kdYaw*yawErrD, -P.maxYawRate, P.maxYawRate);

        [vxBody, px, hist_px] = stepSecondOrderZero(velCmd(1), px, hist_px, P.xy, dt);
        [vyBody, py, hist_py] = stepSecondOrderZero(velCmd(2), py, hist_py, P.xy, dt);
        [vz, pz, hist_pz] = stepFirstOrderDelay(velCmd(3), pz, hist_pz, P.z, dt);
        [yawRate, pyaw, hist_yaw] = stepSecondOrderNoZero(yawRateCmd, pyaw, hist_yaw, P.yaw, dt);

        vWorld = rotYawBodyToWorld(yaw) * [vxBody; vyBody; vz] + windApprox(t(k), windOn);
        X(k+1,1:3) = (pos + vWorld*dt).';
        X(k+1,4:6) = vWorld.';
        X(k+1,7) = wrapToPi(yaw + yawRate*dt);
        X(k+1,8) = yawRate;
        U(k,:) = [velCmd.', yawRateCmd];
        DBG(k,:) = [dhatXY.', norm(windApprox(t(k), windOn))];
    end
end

%% ============================ Dynamics blocks ============================
function [y, x, hist] = stepSecondOrderZero(u, x, hist, par, dt)
    [uD, hist] = delayValue(u, hist, par.L, dt);
    sub = max(1, ceil(dt/0.01)); h = dt/sub;
    for i=1:sub
        dx1 = x(2);
        dx2 = -par.wn^2*x(1) - 2*par.zeta*par.wn*x(2) + par.wn^2*uD;
        x = x + h*[dx1; dx2];
    end
    y = par.K * (x(1) + par.tau_z*x(2));
end

function [y, x, hist] = stepSecondOrderNoZero(u, x, hist, par, dt)
    [uD, hist] = delayValue(u, hist, par.L, dt);
    sub = max(1, ceil(dt/0.002)); h = dt/sub;
    for i=1:sub
        dx1 = x(2);
        dx2 = -par.wn^2*x(1) - 2*par.zeta*par.wn*x(2) + par.wn^2*uD;
        x = x + h*[dx1; dx2];
    end
    y = par.K*x(1);
end

function [y, x, hist] = stepFirstOrderDelay(u, x, hist, par, dt)
    [uD, hist] = delayValue(u, hist, par.L, dt);
    x = x + dt*((par.K*uD - x)/par.tau);
    y = x;
end

function [y, x] = stepSecondOrderLPF(u, x, w, zeta, dt)
    sub = max(1, ceil(dt/0.02)); h = dt/sub;
    for i=1:sub
        dx1 = x(2);
        dx2 = -w^2*x(1) - 2*zeta*w*x(2) + w^2*u;
        x = x + h*[dx1; dx2];
    end
    y = x(1);
end

function [uD, hist] = delayValue(u, hist, L, dt)
    hist = [hist(2:end); u];
    d = L/max(dt,1e-9); n0 = floor(d); frac = d - n0;
    idxNew = max(1, length(hist)-n0);
    idxOld = max(1, idxNew-1);
    uD = (1-frac)*hist(idxNew) + frac*hist(idxOld);
end

%% ============================ Helper functions ============================
function P = defaultParams()
    P.dt = 0.0833;
    P.kp = [0.38; 0.38; 1.18]; P.ki = [0.040; 0.040; 0.60]; P.kd = [0.30; 0.30; 0.96];
    P.kiSat = [0.40; 0.40; 0.58]; P.intLeak = [1.8; 1.8; 1.5];
    P.maxVel = [0.78; 0.78; 0.78]; P.maxXYSpeed = 0.78;
    P.kpYaw = 2.05; P.kiYaw = 0.42; P.kdYaw = 0.16; P.kiYawSat = 0.18;
    P.intYawLeak = 2.0; P.maxYawRate = 1.74533; P.yawAlignTol = 0.167;
    P.velEstAlpha = 0.56; P.cmdXYLpfBeta = 0.70;
    P.xy.K = 0.9791332892264056; P.xy.wn = 1.1688119784531361; P.xy.zeta = 0.42594658984568196; P.xy.tau_z = 0.2052931630745509; P.xy.L = 0.1153442442895058;
    P.z.K = 0.9564045415827129; P.z.tau = 0.1659312509862405; P.z.L = 0.05598021989635007;
    P.yaw.K = 1.0018688634605593; P.yaw.wn = 40.05278984476113; P.yaw.zeta = 0.1799212498074827; P.yaw.L = 0.00014540324314614805;
    P.qW = 0.80; P.qZeta = 0.95; P.kCompXY = 0.80; P.dhatMax = 0.40;
    P.dobZeroEdge = 0.020; P.dobFullEdge = 0.080; P.dTermZeroEdge = 0.010; P.dTermFullEdge = 0.045;
    P.pBoostFullEdge = 0.002; P.pBoostZeroEdge = 0.020; P.pBoostGain = 0.60; P.softDeadzone = 0.004;
end

function w = windApprox(t, on)
    if ~on, w=[0;0;0]; return; end
    base = [0.018*cos(0.35*t + 0.4); 0.014*sin(0.31*t + 1.1); 0];
    gust = [0.010*sin(2*pi*0.22*t)^2; -0.008*sin(2*pi*0.17*t)^2; 0];
    w = base + gust;
end
function R = rotWorldToYawBody(yaw), c=cos(yaw); s=sin(yaw); R=[c s 0; -s c 0; 0 0 1]; end
function R = rotYawBodyToWorld(yaw), c=cos(yaw); s=sin(yaw); R=[c -s 0; s c 0; 0 0 1]; end
function y = wrapToPi(a), y = mod(a+pi, 2*pi)-pi; end
function y = wrapToPiVec(a), y = mod(a+pi, 2*pi)-pi; end
function y = clipScalar(x,lo,hi), y=min(max(x,lo),hi); end
function y = clipVec(x,lo,hi), y=min(max(x,lo),hi); end
function y = smoothstep(x,e0,e1), z=min(max((x-e0)/(e1-e0),0),1); y=z*z*(3-2*z); end
function y = softDeadzone(x,epsVal), y=x*(1-exp(-(x/epsVal)^2)); end
function Gd = padeDelay(L,n), if L<=1e-9, Gd=1; else, [num,den]=pade(L,n); Gd=tf(num,den); end, end

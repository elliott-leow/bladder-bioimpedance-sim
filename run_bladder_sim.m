%RUN_BLADDER_SIM  Main entry point for the EIDORS bladder bioimpedance simulation.
%
%   Prerequisites:
%     1. EIDORS installed and on MATLAB path:
%        run('/path/to/eidors/startup.m');
%     2. Netgen available (bundled with EIDORS, or install separately)
%
%   This script runs the simulation steps in order:
%     Step 1: Build and visualise a single model
%     Step 2: Sensitivity analysis across volumes
%     Step 3: Electrode placement optimisation (slow — skip for quick test)
%     Step 4: Multi-frequency sweep

%% --- Ensure MATLAB can find Homebrew binaries (wine, netgen) ---
if isempty(strfind(getenv('PATH'), '/opt/homebrew/bin'))
    setenv('PATH', ['/opt/homebrew/bin:' getenv('PATH')]);
end

%% --- Verify EIDORS ---
if ~exist('mk_image', 'file')
    error(['EIDORS is not on the MATLAB path.\n' ...
           'Run:  run(''/path/to/eidors/startup.m'')  first.']);
end
fprintf('EIDORS detected. Starting bladder simulation...\n\n');

%% --- Step 1: Build and visualise a single model ---
fprintf('===== STEP 1: Build model at 300 mL =====\n');
[fmdl, img] = build_pelvis_model(300);

figure('Name', 'Pelvis Model - 300 mL');
show_fem(img, [1, 1]);
title('Tissue conductivity map (300 mL bladder)');

% Quick forward solve test
data = fwd_solve(img);
fprintf('Forward solve OK. Measurements (%d channels):\n', length(data.meas));
for k = 1:length(data.meas)
    fprintf('  Ch%d: %.6f V\n', k, data.meas(k));
end

%% --- Step 2: Sensitivity analysis ---
fprintf('\n===== STEP 2: Sensitivity analysis =====\n');
run_sensitivity_analysis();

%% --- Step 3: Electrode placement optimisation (optional, slow) ---
% Uncomment the following to run the full sweep (~100 configs x 3 solves each).
% fprintf('\n===== STEP 3: Electrode placement sweep =====\n');
% optimize_electrode_placement();

%% --- Step 4: Multi-frequency sweep ---
fprintf('\n===== STEP 4: Frequency sweep =====\n');
frequency_sweep();

fprintf('\n===== Simulation complete =====\n');

function optimize_electrode_placement()
%OPTIMIZE_ELECTRODE_PLACEMENT  Sweep electrode positions to find the
%   configuration that maximizes bladder sensitivity ratio.

    %% --- Sweep ranges ---
    lower_row_z = [4, 5, 6, 7, 8];          % cm above pubic bone
    upper_row_z = [12, 13, 14, 15];          % cm above pubic bone
    lateral_deg = [10, 15, 20, 25, 30];      % degrees from midline

    n_configs = length(lower_row_z) * length(upper_row_z) * length(lateral_deg);
    % Columns: [lower_z, upper_z, lat_deg, sens_ratio, dZ_per_mL]
    results = NaN(n_configs, 5);
    config_idx = 0;

    for lz = lower_row_z
        for uz = upper_row_z
            for lat = lateral_deg
                config_idx = config_idx + 1;
                fprintf('Config %d/%d: lower_z=%d, upper_z=%d, lat=%d deg\n', ...
                    config_idx, n_configs, lz, uz, lat);

                try
                    % Jacobian sensitivity ratio at 300 mL
                    [fmdl, img] = build_pelvis_model_param(300, lz, uz, lat);
                    J = calc_jacobian(img);
                    total_sens = sum(abs(J), 1);

                    is_bl = get_bladder_mask(fmdl, 300);
                    if ~any(is_bl)
                        fprintf('  No bladder elements found, skipping.\n');
                        results(config_idx, 1:3) = [lz, uz, lat];
                        continue;
                    end
                    sens_ratio = mean(total_sens(is_bl)) / mean(total_sens(~is_bl));

                    % Delta-Z between 100 and 500 mL
                    [~, img_lo] = build_pelvis_model_param(100, lz, uz, lat);
                    [~, img_hi] = build_pelvis_model_param(500, lz, uz, lat);

                    data_lo = fwd_solve(img_lo);
                    data_hi = fwd_solve(img_hi);

                    dZ = abs(data_hi.meas(1) - data_lo.meas(1)) / 0.001;
                    dZ_per_mL = dZ / (500 - 100);

                    results(config_idx, :) = [lz, uz, lat, sens_ratio, dZ_per_mL];
                    fprintf('  Sensitivity ratio: %.2f, dZ/dV: %.6f Ohm/mL\n', ...
                        sens_ratio, dZ_per_mL);

                catch ME
                    fprintf('  FAILED: %s\n', ME.message);
                    results(config_idx, 1:3) = [lz, uz, lat];
                end
            end
        end
    end

    %% --- Find optimal ---
    valid = ~isnan(results(:, 4));
    if ~any(valid)
        fprintf('\nNo valid configurations found.\n');
        return;
    end
    valid_results = results(valid, :);
    [~, best_idx] = max(valid_results(:, 4));

    fprintf('\n========================================\n');
    fprintf('OPTIMAL ELECTRODE CONFIGURATION:\n');
    fprintf('  Lower row:      %.0f cm above pubic bone\n', valid_results(best_idx, 1));
    fprintf('  Upper row:      %.0f cm above pubic bone\n', valid_results(best_idx, 2));
    fprintf('  Lateral offset: %.0f degrees\n',             valid_results(best_idx, 3));
    fprintf('  Sensitivity ratio: %.2f\n',                  valid_results(best_idx, 4));
    fprintf('  dZ/dV: %.6f Ohm/mL\n',                      valid_results(best_idx, 5));
    fprintf('========================================\n');

    %% --- Visualize ---
    figure('Name', 'Electrode Placement Optimization');

    best_uz = valid_results(best_idx, 2);
    mask = valid_results(:, 2) == best_uz;
    subset = valid_results(mask, :);

    scatter3(subset(:,1), subset(:,3), subset(:,4), 100, subset(:,4), 'filled');
    xlabel('Lower Row Z (cm)');
    ylabel('Lateral Offset (deg)');
    zlabel('Sensitivity Ratio');
    title(sprintf('Sensitivity vs Placement (Upper Row = %d cm)', best_uz));
    colorbar;
    grid on;
end

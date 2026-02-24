function run_sensitivity_analysis()
%RUN_SENSITIVITY_ANALYSIS  Compute impedance vs bladder volume and
%   Jacobian sensitivity maps to evaluate electrode placement quality.

    %% --- Forward solve at multiple bladder volumes ---
    volumes = [50, 100, 200, 300, 400, 500];  % mL
    n_ch = 4;  % 3 meas from stim1 + 1 from stim2
    V_meas = zeros(length(volumes), n_ch);

    for i = 1:length(volumes)
        fprintf('\n=== Bladder volume: %d mL ===\n', volumes(i));
        [fmdl, img] = build_pelvis_model(volumes(i));

        data = fwd_solve(img);
        n_meas = length(data.meas);
        if n_meas ~= n_ch
            warning('Expected %d measurements, got %d', n_ch, n_meas);
            n_ch = min(n_ch, n_meas);
        end
        V_meas(i, 1:n_ch) = data.meas(1:n_ch)';
        fprintf('Measured voltages:');
        fprintf(' %.6f', data.meas(1:n_ch));
        fprintf(' V\n');
    end

    %% --- Impedance change per mL ---
    Z = V_meas(:, 1:n_ch) / 0.001;  % V / I -> Ohms

    ch_names = {'Ch1: Sagittal (E2-E4)', 'Ch2: Right Lat (E5-E7)', ...
                'Ch3: Left Lat (E6-E8)',  'Ch4: Transverse (E6-E8 cross)'};
    colors = {'b', 'r', 'g', 'm'};

    figure('Name', 'Impedance vs Bladder Volume');
    for ch = 1:n_ch
        subplot(2, 2, ch);
        plot(volumes, Z(:, ch), [colors{ch} 'o-'], 'LineWidth', 2);
        xlabel('Bladder Volume (mL)');
        ylabel('Impedance (Ohm)');
        title(ch_names{ch});
        grid on;

        p = polyfit(volumes(:), Z(:, ch), 1);
        fprintf('%s: Sensitivity = %.4f Ohm/mL\n', ch_names{ch}, p(1));
    end
    sgtitle('Impedance vs Bladder Volume - All Channels');

    %% --- Jacobian sensitivity map at 300 mL ---
    fprintf('\nComputing Jacobian at 300 mL reference...\n');
    [fmdl_ref, img_ref] = build_pelvis_model(300);

    J = calc_jacobian(img_ref);
    total_sensitivity = sum(abs(J), 1);

    % Visualize
    img_sens = mk_image(fmdl_ref, total_sensitivity);
    figure('Name', 'Sensitivity Map');
    show_fem(img_sens, [1, 1]);
    title('Sensitivity Map: Bright = high sensitivity to conductivity changes');

    %% --- Bladder-specific sensitivity ratio ---
    is_bladder = get_bladder_mask(fmdl_ref, 300);

    if any(is_bladder)
        sens_bladder = mean(total_sensitivity(is_bladder));
        sens_other   = mean(total_sensitivity(~is_bladder));
        ratio = sens_bladder / sens_other;

        fprintf('\n=== Sensitivity Ratio ===\n');
        fprintf('Mean sensitivity (bladder elements): %.6f\n', sens_bladder);
        fprintf('Mean sensitivity (non-bladder):      %.6f\n', sens_other);
        fprintf('Ratio (bladder/wall):                %.2f\n', ratio);
        fprintf('(Higher ratio = better electrode placement)\n');
    else
        fprintf('\nWARNING: No elements found inside bladder region.\n');
        fprintf('The bladder may be outside the mesh or too small to capture.\n');
    end
end

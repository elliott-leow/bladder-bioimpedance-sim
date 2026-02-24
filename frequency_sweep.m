function frequency_sweep()
%FREQUENCY_SWEEP  Simulate at multiple excitation frequencies to find the
%   optimal frequency for maximum bladder sensitivity.
%
%   Tissue conductivities are frequency-dependent (Gabriel et al. 1996).

    freqs_Hz = [1e3, 5e3, 10e3, 25e3, 50e3, 100e3, 200e3, 500e3];
    n_freq = length(freqs_Hz);

    % Conductivity lookup table [S/m] at each frequency
    %             1kHz  5kHz  10kHz 25kHz 50kHz 100kHz 200kHz 500kHz
    sigma.skin   = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.40];
    sigma.fat    = [0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.06];
    sigma.muscle = [0.20, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.50];
    sigma.wall   = [0.12, 0.15, 0.17, 0.19, 0.21, 0.24, 0.28, 0.35];
    sigma.urine  = [1.50, 1.55, 1.60, 1.65, 1.75, 1.80, 1.85, 1.90];
    sigma.bg     = [0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.36, 0.42];

    %% --- Build meshes once (geometry doesn't change with frequency) ---
    fprintf('Building meshes...\n');
    [fmdl_300, ~] = build_pelvis_model(300);
    [fmdl_100, ~] = build_pelvis_model(100);
    [fmdl_500, ~] = build_pelvis_model(500);

    % Pre-compute tissue masks for each mesh
    masks_300 = compute_tissue_masks(fmdl_300, 300);
    masks_100 = compute_tissue_masks(fmdl_100, 100);
    masks_500 = compute_tissue_masks(fmdl_500, 500);

    % Bladder mask for sensitivity ratio
    is_bl = get_bladder_mask(fmdl_300, 300);

    sensitivity_vs_freq = zeros(n_freq, 1);
    dZ_vs_freq          = zeros(n_freq, 1);

    for fi = 1:n_freq
        freq = freqs_Hz(fi);
        fprintf('\n=== Frequency: %.0f kHz ===\n', freq / 1e3);

        % Apply frequency-specific conductivities
        img_300 = apply_sigma(fmdl_300, masks_300, sigma, fi);
        img_100 = apply_sigma(fmdl_100, masks_100, sigma, fi);
        img_500 = apply_sigma(fmdl_500, masks_500, sigma, fi);

        % Jacobian sensitivity ratio at 300 mL
        J = calc_jacobian(img_300);
        total_sens = sum(abs(J), 1);

        if any(is_bl)
            sensitivity_vs_freq(fi) = mean(total_sens(is_bl)) / ...
                                       mean(total_sens(~is_bl));
        end

        % dZ between 100 and 500 mL
        data_lo = fwd_solve(img_100);
        data_hi = fwd_solve(img_500);
        dZ_vs_freq(fi) = abs(data_hi.meas(1) - data_lo.meas(1)) / 0.001 / 400;

        fprintf('Sensitivity ratio: %.2f, dZ/dV: %.6f Ohm/mL\n', ...
            sensitivity_vs_freq(fi), dZ_vs_freq(fi));
    end

    %% --- Plot ---
    figure('Name', 'Frequency Optimization');

    subplot(2, 1, 1);
    semilogx(freqs_Hz / 1e3, sensitivity_vs_freq, 'bo-', 'LineWidth', 2);
    xlabel('Frequency (kHz)');
    ylabel('Bladder Sensitivity Ratio');
    title('Sensitivity Ratio vs Excitation Frequency');
    grid on;

    subplot(2, 1, 2);
    semilogx(freqs_Hz / 1e3, dZ_vs_freq, 'ro-', 'LineWidth', 2);
    xlabel('Frequency (kHz)');
    ylabel('\DeltaZ / \DeltaV (Ohm/mL)');
    title('Impedance Change per mL vs Frequency');
    grid on;

    [~, best_fi] = max(sensitivity_vs_freq);
    fprintf('\nOptimal frequency: %.0f kHz (sensitivity ratio: %.2f)\n', ...
        freqs_Hz(best_fi) / 1e3, sensitivity_vs_freq(best_fi));
end

%% ===== Local helper functions =====

function masks = compute_tissue_masks(fmdl, bladder_volume_mL)
%COMPUTE_TISSUE_MASKS  Return struct of logical masks for each tissue layer.
    torso_rx = 15; torso_ry = 10;
    skin_thick = 0.2; fat_thick = 1.5; muscle_thick = 1.5;

    scale = (bladder_volume_mL / 300)^(1/3);
    bl_a = 2.5 * scale;
    bl_b = 2.0 * scale;
    bl_c = 2.5 * scale;
    bl_wall = 0.3;
    bl_center = [0, -2, 3 + bl_c];

    ec = interp_mesh(fmdl);
    cx = ec(:,1); cy = ec(:,2); cz = ec(:,3);

    r_el = sqrt((cx / torso_rx).^2 + (cy / torso_ry).^2);
    avg_r = (torso_rx + torso_ry) / 2;

    skin_bnd   = 1 - skin_thick / avg_r;
    fat_bnd    = skin_bnd - fat_thick / avg_r;
    muscle_bnd = fat_bnd  - muscle_thick / avg_r;

    masks.skin   = r_el > skin_bnd;
    masks.fat    = r_el > fat_bnd    & r_el <= skin_bnd;
    masks.muscle = r_el > muscle_bnd & r_el <= fat_bnd;

    bx = (cx - bl_center(1)) / bl_a;
    by = (cy - bl_center(2)) / bl_b;
    bz = (cz - bl_center(3)) / bl_c;
    r_bl = sqrt(bx.^2 + by.^2 + bz.^2);
    avg_bl = (bl_a + bl_b + bl_c) / 3;

    masks.urine = r_bl < (1 - bl_wall / avg_bl);
    masks.wall  = r_bl < 1 & ~masks.urine;
end

function img = apply_sigma(fmdl, masks, sigma, fi)
%APPLY_SIGMA  Create an EIDORS image with frequency-specific conductivities.
    img = mk_image(fmdl, sigma.bg(fi));
    img.elem_data(masks.muscle) = sigma.muscle(fi);
    img.elem_data(masks.fat)    = sigma.fat(fi);
    img.elem_data(masks.skin)   = sigma.skin(fi);
    img.elem_data(masks.wall)   = sigma.wall(fi);
    img.elem_data(masks.urine)  = sigma.urine(fi);
end

function [fmdl, img] = build_pelvis_model_param(bladder_volume_mL, lower_z, upper_z, lat_deg)
%BUILD_PELVIS_MODEL_PARAM  Create a 3D FEM pelvis model with configurable
%   electrode positions for bioimpedance simulation.
%
%   Inputs:
%     bladder_volume_mL - bladder fill volume in mL (50-600 typical)
%     lower_z           - z-height of lower electrode row (cm above pubic bone)
%     upper_z           - z-height of upper electrode row (cm above pubic bone)
%     lat_deg           - lateral electrode angular offset (degrees from midline)
%
%   Returns:
%     fmdl - EIDORS forward model struct
%     img  - EIDORS image with conductivity distribution

    if nargin < 1, bladder_volume_mL = 300; end
    if nargin < 2, lower_z = 6; end
    if nargin < 3, upper_z = 14; end
    if nargin < 4, lat_deg = 20; end

    %% --- Geometry parameters (all in cm) ---
    torso_rx = 15;       % lateral half-width
    torso_ry = 10;       % anterior-posterior half-depth
    torso_h  = 20;       % total model height (pubic bone to above navel)

    skin_thick   = 0.2;
    fat_thick    = 1.5;  % adjust for patient habitus (0.5-4 cm)
    muscle_thick = 1.5;

    % Bladder as ellipsoid; scale semi-axes with volume
    scale = (bladder_volume_mL / 300)^(1/3);
    bl_a = 2.5 * scale;   % lateral semi-axis (cm)
    bl_b = 2.0 * scale;   % AP semi-axis (cm)
    bl_c = 2.5 * scale;   % SI semi-axis (cm)
    bl_wall = 0.3;        % detrusor wall thickness (cm)
    bl_center = [0, -2, 3 + bl_c];  % x=0, y=-2 (posterior), z above pubic bone

    %% --- Electrode layout ---
    % 8 electrodes on the anterior abdominal wall.
    % Angle convention: 0 = anterior midline, increases clockwise from above
    %   so sin(angle) -> +x (right), cos(angle) -> +y (anterior)
    n_elec = 8;
    elec_pos = [ ...
    %   angle(deg)            z(cm)
        0,                    lower_z;   % E1: source, anterior midline
        lat_deg/2,            lower_z;   % E2: +V sagittal, slight right
        360 - lat_deg*3/4,    upper_z;   % E3: sink, near midline upper
        360 - lat_deg*5/4,    upper_z;   % E4: -V sagittal, moderate left
        lat_deg,              lower_z;   % E5: +V lateral right, lower
        360 - lat_deg,        lower_z;   % E6: +V lateral left, lower
        lat_deg,              upper_z;   % E7: -V lateral right, upper
        360 - lat_deg*3/2,    upper_z;   % E8: -V lateral left, upper
    ];
    elec_diam = 1.0;  % cm, Ag/AgCl electrode diameter

    %% --- Create the FEM mesh via Netgen ---
    % ng_mk_ellip_models often fails with "Duplicate elements on boundary"
    % for elliptical cross-sections. Workaround: mesh a cylinder with
    % ng_mk_cyl_models (much more robust), then scale nodes to elliptical.
    %
    % max_edge_length: larger = coarser/faster mesh.
    % Use 5.0 for quick testing, 2.0-3.0 for final runs.
    max_edge = 5.0;  % COARSE for fast iteration
    avg_r = (torso_rx + torso_ry) / 2;

    try
        fmdl = ng_mk_cyl_models( ...
            {torso_h, avg_r, max_edge}, ...
            [n_elec, 0.5], ...
            [elec_diam / 2]);
    catch ME1
        fprintf('Cell format failed: %s\nTrying vector format...\n', ME1.message);
        fmdl = ng_mk_cyl_models( ...
            [torso_h, avg_r], ...
            [n_elec, 0.5], ...
            [elec_diam / 2]);
    end

    % Scale circular cross-section to elliptical
    fmdl.nodes(:,1) = fmdl.nodes(:,1) * (torso_rx / avg_r);
    fmdl.nodes(:,2) = fmdl.nodes(:,2) * (torso_ry / avg_r);

    %% --- Override electrode positions to anatomical locations ---
    % Use find_boundary to get true surface nodes (not interior nodes)
    bdy = find_boundary(fmdl);
    bdy_nodes = unique(bdy(:));

    % Exclude top/bottom cap nodes — electrodes only on the side surface
    zn = fmdl.nodes(bdy_nodes, 3);
    side_mask = zn > 0.5 & zn < (torso_h - 0.5);
    side_nodes = bdy_nodes(side_mask);
    sn_coords  = fmdl.nodes(side_nodes, :);

    for i = 1:n_elec
        ang_rad = elec_pos(i,1) * pi / 180;
        z_tgt   = elec_pos(i,2);

        % Target point on the elliptical surface
        x_tgt = torso_rx * sin(ang_rad);
        y_tgt = torso_ry * cos(ang_rad);

        % Distance from each boundary node to target
        dist = sqrt((sn_coords(:,1) - x_tgt).^2 + ...
                     (sn_coords(:,2) - y_tgt).^2 + ...
                     (sn_coords(:,3) - z_tgt).^2);
        [~, si] = sort(dist);
        nn = min(3, length(si));  % 3 nodes per electrode
        fmdl.electrode(i).nodes    = side_nodes(si(1:nn))';
        fmdl.electrode(i).z_contact = 0.01;  % Ohm*m^2
    end

    %% --- Stimulation patterns ---
    % 1 mA injection current (typical for bioimpedance)
    I_mA = 0.001;

    % Stim 1: Drive E1(+) to E3(-), measure 3 differential pairs
    stim(1).stim_pattern = zeros(n_elec, 1);
    stim(1).stim_pattern(1) =  I_mA;   % E1 source
    stim(1).stim_pattern(3) = -I_mA;   % E3 sink
    stim(1).meas_pattern = zeros(3, n_elec);
    stim(1).meas_pattern(1, [2, 4]) = [ 1, -1];  % Ch1: E2 - E4
    stim(1).meas_pattern(2, [5, 7]) = [ 1, -1];  % Ch2: E5 - E7
    stim(1).meas_pattern(3, [6, 8]) = [ 1, -1];  % Ch3: E6 - E8

    % Stim 2: Drive E5(+) to E7(-), measure 1 pair
    stim(2).stim_pattern = zeros(n_elec, 1);
    stim(2).stim_pattern(5) =  I_mA;
    stim(2).stim_pattern(7) = -I_mA;
    stim(2).meas_pattern = zeros(1, n_elec);
    stim(2).meas_pattern(1, [6, 8]) = [1, -1];   % Ch4: E6 - E8

    fmdl.stimulation = stim;

    %% --- Solver functions ---
    fmdl.solve      = @fwd_solve_1st_order;
    fmdl.system_mat = @system_mat_1st_order;
    fmdl.jacobian   = @jacobian_adjoint;
    fmdl = mdl_normalize(fmdl, 0);  % absolute (not normalized) measurements

    %% --- Create image AFTER fmdl is fully configured ---
    img = mk_image(fmdl, 0.30);  % background = pelvic connective tissue (S/m)

    %% --- Assign tissue conductivities ---
    ec = interp_mesh(fmdl);  % element centroids [n_elem x 3]
    cx = ec(:,1); cy = ec(:,2); cz = ec(:,3);

    % Normalised elliptical distance from center (1.0 = on surface)
    r_el = sqrt((cx / torso_rx).^2 + (cy / torso_ry).^2);

    % Layer boundaries (outside-in), using average radius for thickness scaling
    avg_r = (torso_rx + torso_ry) / 2;
    skin_bnd   = 1 - skin_thick / avg_r;
    fat_bnd    = skin_bnd - fat_thick / avg_r;
    muscle_bnd = fat_bnd  - muscle_thick / avg_r;

    is_skin   = r_el > skin_bnd;
    is_fat    = r_el > fat_bnd    & r_el <= skin_bnd;
    is_muscle = r_el > muscle_bnd & r_el <= fat_bnd;

    % Bladder ellipsoid
    bx = (cx - bl_center(1)) / bl_a;
    by = (cy - bl_center(2)) / bl_b;
    bz = (cz - bl_center(3)) / bl_c;
    r_bl = sqrt(bx.^2 + by.^2 + bz.^2);

    avg_bl_r = (bl_a + bl_b + bl_c) / 3;
    is_urine = r_bl < (1 - bl_wall / avg_bl_r);
    is_wall  = r_bl < 1 & ~is_urine;

    % Conductivities at 50 kHz (S/m) — Gabriel et al. 1996
    img.elem_data(is_muscle) = 0.35;
    img.elem_data(is_fat)    = 0.04;
    img.elem_data(is_skin)   = 0.10;
    img.elem_data(is_wall)   = 0.21;
    img.elem_data(is_urine)  = 1.75;

    fprintf('Model: %d elems, %d nodes | Bladder: %.0f mL (%.1fx%.1fx%.1f cm)\n', ...
        size(fmdl.elems,1), size(fmdl.nodes,1), bladder_volume_mL, bl_a, bl_b, bl_c);
end

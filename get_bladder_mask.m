function is_bladder = get_bladder_mask(fmdl, bladder_volume_mL)
%GET_BLADDER_MASK  Return a logical mask of elements inside the bladder.
%   is_bladder = get_bladder_mask(fmdl, bladder_volume_mL)
%
%   Uses the same geometry conventions as build_pelvis_model_param.

    if nargin < 2, bladder_volume_mL = 300; end

    scale = (bladder_volume_mL / 300)^(1/3);
    bl_a = 2.5 * scale;
    bl_b = 2.0 * scale;
    bl_c = 2.5 * scale;
    bl_center = [0, -2, 3 + bl_c];

    ec = interp_mesh(fmdl);
    bx = (ec(:,1) - bl_center(1)) / bl_a;
    by = (ec(:,2) - bl_center(2)) / bl_b;
    bz = (ec(:,3) - bl_center(3)) / bl_c;

    is_bladder = sqrt(bx.^2 + by.^2 + bz.^2) < 1;
end

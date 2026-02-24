function [fmdl, img] = build_pelvis_model(bladder_volume_mL)
%BUILD_PELVIS_MODEL  Create a 3D FEM pelvis model for bioimpedance simulation.
%   [fmdl, img] = build_pelvis_model(bladder_volume_mL)
%
%   Wrapper around build_pelvis_model_param with default electrode placement:
%     lower row z = 6 cm, upper row z = 14 cm, lateral offset = 20 deg.

    if nargin < 1, bladder_volume_mL = 300; end
    [fmdl, img] = build_pelvis_model_param(bladder_volume_mL, 6, 14, 20);
end

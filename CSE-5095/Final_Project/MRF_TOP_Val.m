clear all; close all; clc;
%% source folders containing scripts not in this folder
addpath(genpath('FE_routines'))
addpath(genpath('functions'))
addpath(genpath('mesh_utilities'))
addpath(genpath('optimization'))
addpath(genpath('utilities'))
addpath(genpath('plotting'))
global OPT FE
OPT.train_input = 'data/train/input/';
OPT.train_output = 'data/train/output/';
OPT.val_input = 'data/val/input/';
OPT.val_output = 'data/val/output/';

path_input = 'data/val/input/';
path_out_1 = 'data/val/output/';
path_out_2 = 'data/val/output_Mises/';
%% Initialization
% ============================
no_input = 10; % number of samples
% Latin Hypercube Sampling in [0,1]
% Scale to your parameter ranges
TR_min_List = 30*rand(5);        % scale to [0,30]
% Display results
for index = 1:no_input
    OPT.TR_min = TR_min_List(index);
    OPT.TR_max = OPT.TR_min  + 5;
    get_inputs();
    OPT.options.max_iter = 100;
    init_FE();
    init_optimization();
    %% Analysis
    perform_analysis();
    compute_stress_U();
    F = FE.elem_node'; % matrix of faces to be sent to patch function
    V = FE.coords'; % vertex list to be sent to patch function
    % -------------------------------------
    % fig_eps_x = myfig(2,F,V,FE.strain(1,:));
    % fig_eps_y = myfig(3,F,V,FE.strain(2,:));
    % fig_eps_xy = myfig(4,F,V,FE.strain(3,:));
    % -------------------------------------
    %% Optimization
    runmma(OPT.dv, @(x)obj(x), @(x)nonlcon(x));
    % fig_TOP = myfig(5,F,V, OPT.pen_rho_e);
    % figMises = myfig(6,F,V,FE.svm);
    % targetSize = [256,256]; % resize to 128x128
    % Suppose FE.strain(1,:), FE.strain(2,:), FE.strain(3,:) are your strain components
    a1 = reshape(FE.strain(1,:), [80, 80]);
    a2 = reshape(FE.strain(2,:), [80, 80]);
    a3 = reshape(FE.strain(3,:), [80, 80]);
    % Combine them into one 3-channel array
    X = cat(3, a1, a2, a3);   % size: 80 x 80 x 3
    % Optionally normalize
    X = (X - min(X(:))) / (max(X(:)) - min(X(:)));   % normalize to [0,1]
    Y = reshape(OPT.pen_rho_e, [80, 80]);
    Y_Mises = reshape(FE.svm, [80, 80]);
    % Normalize density (Y) to [0,1]
    Y = (Y - min(Y(:))) / (max(Y(:)) - min(Y(:)));
    % Normalize von Mises stress (Y_Mises) to [0,1]
    Y_Mises = (Y_Mises - min(Y_Mises(:))) / (max(Y_Mises(:)) - min(Y_Mises(:)));
    % Save to .mat file for later Python training
    save(fullfile(path_input, ['input_', num2str(index), '.mat']), 'X');
    save(fullfile(path_out_1, ['output_', num2str(index), '.mat']), 'Y');
    save(fullfile(path_out_2, ['output_Mises_', num2str(index), '.mat']), 'Y_Mises');
    % % ================================
end
%

save('Lbracket2d.mat', 'F', 'V');




% Instructions:
%
% 1)  Install the NFFT3 library from [1] and configure the MATLAB 
%     interface. Make sure the directories nfft/matlab/nfft and
%     nfft/matlab/fastsum are on the MATLAB path.
%
% 2)  Download and unpack the NFFT-Lanczos example code from [2]. Make sure
%     the files "fastsumAdajcencyEigs" and "fastsumAdjacencySetup" are on
%     the MATLAB path.
%     
% 3a) Download and unpack the Oakland "training" point cloud from [3].
%     Rename the file "training\oakland_part3_an_training.xyz_label_conf"
%     to "Oakland1.dat" and save it in this directory.
% 3b) Download and unpack the Oakland "validation" point cloud from [4].
%     Rename the file "validation\oakland_part3_am.xyz_label_conf" to
%     "Oakland2.dat" and save it in this directory.
%
% 4)  In order to create the files required for running a specific
%     experiment, set the variables "dataset", "sigma", and "num_ev" in the
%     first section of this script, then run this script.
%
% Links:
% [1] https://github.com/NFFT/nfft
% [2] https://www.tu-chemnitz.de/mathematik/wire/people/files_alfke/NFFT-Lanczos-Example-v1.tar.gz
% [3] http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/data/training.zip
% [4] http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/data/validation.zip


%% Specify dataset, sigma, and number of eigenvalues

% dataset: either 'Oakland1' or 'Oakland2'
dataset = 'Oakland1';

% sigma: value for sigma in Gaussian kernel
sigma = 100;

% num_ev: number of computed eigenvalues. Enables pseudoinverse
% approximation with rank up to num_ev-1.
num_ev = 51;


%% Load data

input_file = sprintf('%s.dat', dataset);
data = table2array(readtable(input_file));
x = data(:,1:3);
[~,~,y] = unique(data(:,4));


%% Setup NFFT-Fastsum

opts.doNormalize = 1;
opts.sigma = sigma;
opts.diagonalEntry = 0;
opts.N = 32;
opts.m = 4;
opts.p = 1;
opts.eps_B = 0;
opts.eigs_tol = 1e-3;


%% Compute eigenvalues

ticEV = tic;

[U, Lambda, S] = fastsumAdjacencyEigs(x, num_ev, opts);
w = 1-diag(Lambda);

eigenvalue_time = toc(ticEV);
fprintf('Eigenvalue computation took %g s \n', eigenvalue_time);


%% Save result

output_file = sprintf('%s_sigma%g.mat', dataset, sigma);
save(output_file, 'x', 'y', 'w', 'U', 'eigenvalue_time', 'dataset', 'sigma', 'num_ev', 'input_file');
fprintf('Results saved to file %s \n', output_file)

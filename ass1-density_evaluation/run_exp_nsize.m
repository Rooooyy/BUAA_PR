clc;clear;
global train_size;
global test_size;
train_size = 3200;   % 100 200 400 800 1600 3200
test_size = train_size / 2;

% generate_sample;
% main_square_parzen;

generate_sample;
main_gauss_parzen;

% generate_sample;
% main_sphere_parzen;

% generate_sample;
% main_knn;

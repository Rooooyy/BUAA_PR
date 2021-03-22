clc;clear;
%%%%%%%%%%
% 混合高斯分布的参数
%%%%%%%%%%
p = [0.3 0.7];
mu = [-3 5; 8 -8];
sigma = cat(3, [5 5], [6 6]); % 第一个参数表示concat的维度，dim=3就是把数据stack起来
%%%%%%%%%%
% 生成样本
% 生成train_size个点，估计test_size个点的密度
%%%%%%%%%%
global train_size;
%train_size = 2000;
global test_size;
%test_size = train_size/2;

gm = gmdistribution(mu,sigma,p);

global X_train;
[X_train, ~] = random(gm,train_size);

global X_test;
[X_test, ~] = random(gm,test_size);

global P_test;
P_test = pdf(gm,X_test);  % pdf函数可以返回指定分布在X处的真实值

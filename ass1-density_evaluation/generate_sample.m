clc;clear;
%%%%%%%%%%
% ��ϸ�˹�ֲ��Ĳ���
%%%%%%%%%%
p = [0.3 0.7];
mu = [-3 5; 8 -8];
sigma = cat(3, [5 5], [6 6]); % ��һ��������ʾconcat��ά�ȣ�dim=3���ǰ�����stack����
%%%%%%%%%%
% ��������
% ����train_size���㣬����test_size������ܶ�
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
P_test = pdf(gm,X_test);  % pdf�������Է���ָ���ֲ���X������ʵֵ

function p_prediction = gauss_parzen_predict( x_)
% ��˹��
% ���Ƹ����ܶȺ���ֵ p=k/n/V 
% xiΪ������,hΪ���ڿ��
% k�ļ��㣺����ʾ�Ժ������ phi(xi)/V = gauss(X,mu,sigma)��muΪ�����㣬sigmaΪ�������
% �����ܶ�p_prediction = 1/n*sigma(phi(xi)/V)
% `����gauss�������sigma����
global train_size;
global X_train;
global h_glb;
xx = cell(train_size,1);
det_h_ssigma = cell(train_size,1);
inv_ssigma = cell(train_size,1); 
sigma = eye(2);  % ����ֱ���õ�λ����Q��Э�������=h^2*Q�����ַ���Ӧ�ö�Բ�ĸ�˹�ֲ��Ƚ����ã���ĸ�˹���ܲ�̫�У�����Ҳû˵���Q��ô�裬���������ƻ���ѧϰ�ķ���ѧϰ���sigma����֪����û�з�ѧϰ���㷨
det_h_sigma = det(sigma);
inv_sigma = inv(sigma);
for i=1:train_size
    xx{i} = x_;
    det_h_ssigma{i} = det_h_sigma;
    inv_ssigma{i} = inv_sigma;
end
phi = cellfun(@gauss_kernel, xx, num2cell(X_train,2), det_h_ssigma, inv_ssigma); % phi(xi)/V = gauss(X,mu,sigma)
p_prediction = sum(phi)/train_size;
end


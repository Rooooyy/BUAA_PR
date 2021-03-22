function p_prediction = sphere_parzen_predict( x_)
% ��
% ���Ƹ����ܶȺ���ֵ
% xiΪ������,hΪ���ڿ��
% k�ļ��㣺����ʾ�Ժ������ phi(xi) = (xi-x_)'*(xi-x_)/h^2<0.5^2
% �����ܶ�p_prediction = 1/n*sigma(phi(xi)/V)
global train_size;
global X_train;
global h_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % �ѵ�������������ÿһ��ѵ��������������
tmp2 = sum(tmp1.^2,2); % ||xxi||^2  ���2����
k = sum(tmp2/h_glb^2<0.25); % I(||xxi||^2/h^2<0.5^2) ��k
p_prediction = k/train_size/(h_glb^2 * pi * 0.25); % 2ά��Բ�����
end


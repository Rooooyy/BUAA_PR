function p_prediction = knn_predict( x_)
% ���Ƹ����ܶȺ���ֵ
% xiΪ������,hΪ���ڿ��
% h�ļ��㣺������x_�ľ��룬��k��ǡ��Ϊ(parzen window�е�)���ڿ��
% �����ܶ�p_prediction = 1/n*k/h^2
global train_size;
global X_train;
global k_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % xxi = xi-x_
tmp2 = sum(tmp1.^2,2); % ||xxi||^2
tmp3 = sort(tmp2, 'ascend');  % ȡǰk����С�Ķ�����
h = sqrt(tmp3(k_glb))*2;  % ���ڵĳ��Ⱦ�����Ϊ��k��neighbour�����ĵľ���*2��������
p_prediction = k_glb/train_size/h^2;

end


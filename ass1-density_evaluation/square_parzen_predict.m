function p_prediction = square_parzen_predict( x_)
% ����
% ���Ƹ����ܶȺ���ֵ
% xiΪ������,hΪ���ڿ��
% k�ļ��㣺����ʾ�Ժ������ phi(xi) = (xi-x_)'*(xi-x_)/h^2<0.5^2
% �����ܶ�p_prediction = 1/n*sigma(phi(xi)/V)������V=h^2
global train_size;
global X_train;
global h_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % �ѵ�������������ÿһ��ѵ��������������
% ���ﲻ����2-������2-��������״����������Ƿ���
% ֱ�ӽ�1-������ӣ�ʹ��С�� d * 0.5h���������Χ�ɵ���������y=|x|,������һ����ƽ�����������������
% �����������������Ļ�Ҳ���ԣ����Ǵ��ڵ�����Ͳ�����h^2����
% ��ȷ������ÿһ��ά�ȵ����Ƚ�
tmp2 = sum(abs(tmp1) < (0.5 * h_glb), 2); % abs(tmp1)��1����, Ҫ��ÿһά���϶���0.5h���Ƚ�
k = sum(tmp2 >= 2);  % ����ά�����=2��ʾ����ά�ȶ�<0.5h
p_prediction = k/train_size/h_glb^2;
end


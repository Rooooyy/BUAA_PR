function p_prediction = sphere_parzen_predict( x_)
% 球窗
% 估计概率密度函数值
% xi为列向量,h为窗口宽度
% k的计算：各点示性函数求和 phi(xi) = (xi-x_)'*(xi-x_)/h^2<0.5^2
% 估计密度p_prediction = 1/n*sigma(phi(xi)/V)
global train_size;
global X_train;
global h_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % 把单个测试样本和每一个训练集样本都作差
tmp2 = sum(tmp1.^2,2); % ||xxi||^2  差的2范数
k = sum(tmp2/h_glb^2<0.25); % I(||xxi||^2/h^2<0.5^2) 求k
p_prediction = k/train_size/(h_glb^2 * pi * 0.25); % 2维，圆的面积
end


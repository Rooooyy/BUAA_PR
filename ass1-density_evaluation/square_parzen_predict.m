function p_prediction = square_parzen_predict( x_)
% 方窗
% 估计概率密度函数值
% xi为列向量,h为窗口宽度
% k的计算：各点示性函数求和 phi(xi) = (xi-x_)'*(xi-x_)/h^2<0.5^2
% 估计密度p_prediction = 1/n*sigma(phi(xi)/V)，其中V=h^2
global train_size;
global X_train;
global h_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % 把单个测试样本和每一个训练集样本都作差
% 这里不能用2-范数，2-范数的形状是球体而不是方形
% 直接将1-范数相加，使其小于 d * 0.5h，这个函数围成的区域类似y=|x|,而不是一个边平行于坐标轴的正方形
% 如果用上面这个方法的话也可以，但是窗口的面积就不能用h^2来算
% 正确做法是每一个维度单独比较
tmp2 = sum(abs(tmp1) < (0.5 * h_glb), 2); % abs(tmp1)是1范数, 要在每一维度上都和0.5h作比较
k = sum(tmp2 >= 2);  % 两个维度相加=2表示两个维度都<0.5h
p_prediction = k/train_size/h_glb^2;
end


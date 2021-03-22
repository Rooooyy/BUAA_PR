function p_prediction = knn_predict( x_)
% 估计概率密度函数值
% xi为列向量,h为窗口宽度
% h的计算：样本到x_的距离，第k个恰好为(parzen window中的)窗口宽度
% 估计密度p_prediction = 1/n*k/h^2
global train_size;
global X_train;
global k_glb;
tmp1 = (X_train-repmat(x_,train_size,1)); % xxi = xi-x_
tmp2 = sum(tmp1.^2,2); % ||xxi||^2
tmp3 = sort(tmp2, 'ascend');  % 取前k个最小的二范数
h = sqrt(tmp3(k_glb))*2;  % 窗口的长度就设置为第k个neighbour到中心的距离*2（方窗）
p_prediction = k_glb/train_size/h^2;

end


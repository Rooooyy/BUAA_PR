function p_prediction = gauss_parzen_predict( x_)
% 高斯窗
% 估计概率密度函数值 p=k/n/V 
% xi为列向量,h为窗口宽度
% k的计算：各点示性函数求和 phi(xi)/V = gauss(X,mu,sigma)，mu为样本点，sigma为样本方差。
% 估计密度p_prediction = 1/n*sigma(phi(xi)/V)
% `可用gauss函数替代sigma内容
global train_size;
global X_train;
global h_glb;
xx = cell(train_size,1);
det_h_ssigma = cell(train_size,1);
inv_ssigma = cell(train_size,1); 
sigma = eye(2);  % 这里直接用单位阵做Q，协方差矩阵=h^2*Q，这种方法应该对圆的高斯分布比较有用，扁的高斯可能不太行，书上也没说这个Q怎么设，除了用类似机器学习的方法学习这个sigma，不知道有没有非学习的算法
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


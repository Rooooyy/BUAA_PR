%%%%%%%%%%
% 高斯parzen window 估计概率密度
%%%%%%%%%%
global train_size;
global test_size;
global X_test;
global P_test;

h = 0.5:0.2:5;
global h_glb;
h_glb = h(1);

mse = zeros(size(h,2),1);
for i=1:size(h,2)
    h_glb = h(i);
    result = cellfun(@gauss_parzen_predict,num2cell(X_test,2));
    mse(i) = sum((P_test-result).^2)/test_size;
end
h_best = h(find(mse==min(mse)));
h_best = h_best(1);
h_glb = h_best;
result = cellfun(@gauss_parzen_predict,num2cell(X_test,2));
mse_best = sum((P_test-result).^2)/test_size;

h_best
mse_best

%%%%%%%%%%
% 画图
%%%%%%%%%%
figure;
suptitle(['train\_size=',num2str(train_size),', test\_size=',num2str(test_size)]);
x = -20:1:20;
y = -20:1:20;
[X,Y] = meshgrid(x,y); % 产生网格数据并处理
p_true = reshape(pdf(gm,[X(:) Y(:)]),size(X));  % 求取真实的联合概率密度
%%%%%% 原始概率密度 %%%%%%
subplot(2,2,1);
surf(X,Y,p_true); % 真实密度函数
axis([-20 20 -20 20 0 0.02]);
hold on;
shading interp;  %对图像平滑处理

%scatter3(X_test(:,1), X_test(:,2), P_test, 'filled', 'MarkerFaceColor', color1); % 样本点

title('原始混合高斯概率密度');

%%%%%% 估计概率密度 %%%%%%
subplot(2,2,2);
Z = griddata(X_test(:,1),X_test(:,2),result,X, Y);
surf(X,Y,Z);
axis([-20 20 -20 20 0 0.04]);
hold on;
shading interp;  %对图像平滑处理
%scatter3(X_test(:,1), X_test(:,2), result, 'filled', 'MarkerFaceColor', color2);
title('parzen窗口法（高斯窗）估计概率密度');

% 有真实值和预测值两组3维数据，所以要分别指定属性
% 分别指定颜色
color1 = [0.1 0.1 0.1];  % 淡黑色
color2 = [0.9 0 0];  % 淡红色
% 散点大小
s1 = repmat([10],test_size,1);
s2 = repmat([20],test_size,1);
s = [s1;s2];
c1 = repmat(color1,test_size,1);
c2 = repmat(color2,test_size,1);
c = [c1;c2];
% 数据，X和Y轴直接复制两份就行，Z轴是概率，一个是真实值一个是估计值
% XX = [X_test(:,1);X_test(:,1)];
% YY = [X_test(:,2);X_test(:,2)];
% ZZ = [P_test;result];

%%%%%% 样本散点&估计散点 %%%%%%
subplot(2,2,3);
%scatter3(XX, YY, ZZ, s, c, 'filled'); % 样本+预测
scatter3(X_test(:,1),X_test(:,2),P_test,s1,c1,'filled');
hold on;  % hold on必须放在这，否则这个图就会变成2维图
scatter3(X_test(:,1),X_test(:,2),result,s2,c2,'filled')
legend('真实值','估计值');
title('估计散点与真实散点');
hold off;

%%%%%% 参数-误差曲线 %%%%%%
subplot(2,2,4);
plot(h,mse,'-k');
hold on;
scatter(h_best, mse_best, 100, color2, 'filled');
legend('误差','误差最低点');
xlabel('窗口大小h');
ylabel('均方误差mse');
title('窗口大小-误差曲线');

hold off;

% 保存图像和数据
set(gcf,'position',[0,0,1200,800]);
f = getframe(gcf);
path = ['res','/gauss_parzen'];
file_name = ['/gauss_parzen_',num2str(train_size),'_',num2str(test_size),'.png'];
if ~exist(path,'dir')
    mkdir(path);
end
imwrite(f.cdata,[path, file_name]);
file_name = ['/gauss_parzen_',num2str(train_size),'_',num2str(test_size),'.txt'];
save([path,file_name],'mse_best','h_best','-ascii');
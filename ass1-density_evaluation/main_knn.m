%%%%%%%%%%
% knn ���Ƹ����ܶ�
%%%%%%%%%%
global train_size;
global test_size;
global X_test;
global P_test;


global k_step;
k_step = 2;
k = 5:k_step:min(train_size,200);
global k_glb;
k_glb = k(1);

mse = zeros(size(k,2),1);
for i=1:size(k,2)
    k_glb = k(i);
    result = cellfun(@knn_predict,num2cell(X_test,2));
    mse(i) = sum((P_test-result).^2)/test_size;
end
k_best = k(find(mse==min(mse)));
k_best = k_best(1);
k_glb = k_best;
result = cellfun(@knn_predict,num2cell(X_test,2));
mse_best = sum((P_test-result).^2)/test_size;
%%%%%%%%%%
% ��ͼ
%%%%%%%%%%

k_best
mse_best

figure;
suptitle(['train\_size=',num2str(train_size),', test\_size=',num2str(test_size)]);
x = -20:1:20;
y = -20:1:20;
[X,Y] = meshgrid(x,y); % �����������ݲ�����
P_true = reshape(pdf(gm,[X(:) Y(:)]),size(X));  % ��ȡ���ϸ����ܶ�
%%%%%% ԭʼ�����ܶ� %%%%%%
subplot(2,2,1);
surf(X,Y,P_true); % ��ʵ�ܶȺ���
axis([-20 20 -20 20 0 0.02]);
hold on;
shading interp;
%scatter3(X_test(:,1), X_test(:,2), P_test, 'filled', 'MarkerFaceColor', color1); % ������
title('ԭʼ��ϸ�˹�����ܶ�');

%%%%%% ���Ƹ����ܶ� %%%%%%
subplot(2,2,2);
Z = griddata(X_test(:,1),X_test(:,2),result,X, Y);
surf(X,Y,Z);
axis([-20 20 -20 20 0 0.02]);
hold on;
%scatter3(X_test(:,1), X_test(:,2), result, 'filled', 'MarkerFaceColor', color2);
shading interp;  %��ͼ��ƽ������
title('knn���Ƹ����ܶ�');

% ��Parzen�Ĵ���ͬ��
color1 = [0.1 0.1 0.1];  % ����
color2 = [0.9 0 0];  % ����
s1 = repmat([10],test_size,1);
s2 = repmat([20],test_size,1);
s = [s1;s2];
c1 = repmat(color1,test_size,1);
c2 = repmat(color2,test_size,1);
c = [c1;c2];
% XX = [X_test(:,1);X_test(:,1)];
% YY = [X_test(:,2);X_test(:,2)];
% ZZ = [P_test;result];

%%%%%% ����ɢ��&����ɢ�� %%%%%%
subplot(2,2,3);
%scatter3(XX, YY, ZZ, s, c, 'filled'); % ����+Ԥ��
scatter3(X_test(:,1),X_test(:,2),P_test,s1,c1,'filled');
hold on;  % hold on��������⣬�������ͼ�ͻ���2άͼ
scatter3(X_test(:,1),X_test(:,2),result,s2,c2,'filled')
legend('��ʵֵ','����ֵ');
title('����ɢ������ʵɢ��');


%%%%%% ����-������� %%%%%%
subplot(2,2,4);
plot(k,mse,'-k');
hold on;
scatter(k_best, mse_best, 100, color2, 'filled');
legend('���','�����͵�');
xlabel('�ڽ�����k');
ylabel('�������mse');
title('�ٽ�����k-�������');

hold off;

% ����ͼ�������
set(gcf,'position',[0,0,1200,800]);
f = getframe(gcf);
path = ['res','/knn'];
file_name = ['/knn_',num2str(train_size),'_',num2str(test_size),'.png'];
if ~exist(path,'dir')
    mkdir(path);
end
imwrite(f.cdata,[path, file_name]);
file_name = ['/knn_',num2str(train_size),'_',num2str(test_size),'.txt'];
save([path,file_name],'mse_best','k_best','-ascii');
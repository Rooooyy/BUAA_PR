%%%%%%%%%%
% ��˹parzen window ���Ƹ����ܶ�
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
% ��ͼ
%%%%%%%%%%
figure;
suptitle(['train\_size=',num2str(train_size),', test\_size=',num2str(test_size)]);
x = -20:1:20;
y = -20:1:20;
[X,Y] = meshgrid(x,y); % �����������ݲ�����
p_true = reshape(pdf(gm,[X(:) Y(:)]),size(X));  % ��ȡ��ʵ�����ϸ����ܶ�
%%%%%% ԭʼ�����ܶ� %%%%%%
subplot(2,2,1);
surf(X,Y,p_true); % ��ʵ�ܶȺ���
axis([-20 20 -20 20 0 0.02]);
hold on;
shading interp;  %��ͼ��ƽ������

%scatter3(X_test(:,1), X_test(:,2), P_test, 'filled', 'MarkerFaceColor', color1); % ������

title('ԭʼ��ϸ�˹�����ܶ�');

%%%%%% ���Ƹ����ܶ� %%%%%%
subplot(2,2,2);
Z = griddata(X_test(:,1),X_test(:,2),result,X, Y);
surf(X,Y,Z);
axis([-20 20 -20 20 0 0.04]);
hold on;
shading interp;  %��ͼ��ƽ������
%scatter3(X_test(:,1), X_test(:,2), result, 'filled', 'MarkerFaceColor', color2);
title('parzen���ڷ�����˹�������Ƹ����ܶ�');

% ����ʵֵ��Ԥ��ֵ����3ά���ݣ�����Ҫ�ֱ�ָ������
% �ֱ�ָ����ɫ
color1 = [0.1 0.1 0.1];  % ����ɫ
color2 = [0.9 0 0];  % ����ɫ
% ɢ���С
s1 = repmat([10],test_size,1);
s2 = repmat([20],test_size,1);
s = [s1;s2];
c1 = repmat(color1,test_size,1);
c2 = repmat(color2,test_size,1);
c = [c1;c2];
% ���ݣ�X��Y��ֱ�Ӹ������ݾ��У�Z���Ǹ��ʣ�һ������ʵֵһ���ǹ���ֵ
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
hold off;

%%%%%% ����-������� %%%%%%
subplot(2,2,4);
plot(h,mse,'-k');
hold on;
scatter(h_best, mse_best, 100, color2, 'filled');
legend('���','�����͵�');
xlabel('���ڴ�Сh');
ylabel('�������mse');
title('���ڴ�С-�������');

hold off;

% ����ͼ�������
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
function p = gauss_kernel( x_, mu, det_h_sigma, inv_sigma )
% ��˹�˺���
% ά��d
global h_glb;
d = 2;
xx = x_-mu;
% p = exp(-xx*inv_sigma*xx'/2/h_glb^2)/(2*pi)^(d/2)/det_h_sigma/h_glb;
p = exp(-xx*inv_sigma*xx'/2/h_glb^2)/(2*pi)^(d/2)/det_h_sigma^(1/2)/h_glb^d;
end


function [tfr] = IMSST_W(x,hlength,num)
% Enhance the time-varying characteristics of the weak components of the signal x using synchrosqueezing operator (SSO).
% INPUT
%    x          :  Signal needed to be column vector.
%    hlength    :  The length of window function.
%    num        :  Iteration number.
% OUTPUT
%    Ts         :  The SSO
%    tfr        :  The STFT

[xrow,xcol] = size(x);

if (xcol~=1)
    error('X must be column vector');
end

if (nargin == 2)
    num=10;
end

if (nargin == 1)
    num=10;
    hlength=xrow/8;
end

hlength=hlength+1-rem(hlength,2);
%ht一條直線　1-27563
ht = linspace(-0.5,0.5,hlength);ht=ht';

% Gaussian window
h = exp(-pi/0.32^2*ht.^2);
[hrow,~]=size(h); 
Lh=(hrow-1)/2;
N=xrow;
t=1:xrow;

tfr= zeros (round(N/2),N) ;

%Compute STFT
for icol=1:N
    % t: time sequence
    % ti: start point
    ti= t(icol); 
    % 
    tau=-min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti]);
    % rem: 取余
    indices= rem(N+tau,N)+1;
    rSig = x(ti+tau,1);
    % .* 两个矩阵每个对应位置元素相乘形成的一个新矩阵
    % conj函数：用于计算复数的共轭值
    % Lh窗的一半
    tfr(indices,icol)=rSig.*conj(h(Lh+1+tau));
end

tfr=fft(tfr);
tfr=tfr(1:round(N/2),:);
end
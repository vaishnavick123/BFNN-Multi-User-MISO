%%% Written by Arpita Thakre June 2, 2021
clear all
close all
clc
%%% Channel generation for MU_MISO Downlink scenario
%%% separation between antenna elements: d = lambda/2

 
K = 2;  %%%  no of downlink users
M = 1;  % no of antennas in each downlink user
N = 64;  % no of antennas at base station
NRF = K; %%% no of RF chains
R = 100000; % no of realizations

 

%% we are generating channel for one downlink user at a time
%% for K users, the code has to be run K times
rng shuffle;
H = zeros(R,N);
los = 1;
non_los = 2;

 

for r = 1:R
    alpha(1) = sqrt(1/2)*(randn(1,los) + sqrt(-1)*randn(1,los));
    alpha(2:non_los+1) = sqrt(0.3162/2)*(randn(1,non_los) + sqrt(-1)*randn(1,non_los)) ; % non_los paths are 5 dB below los path , 10^-0.5
    temp_sum = zeros(1,N);
    
    phi_r = 2*pi*rand(1,(los+non_los));
    phi_t = 2*pi*rand(1,(los+non_los));
    for ele = 1:(los + non_los)
        a_r(:,ele) = sqrt(1/M)*exp(sqrt(-1)*pi*sin(phi_r(ele))*(0:M-1));  %%% array response vector at receiver
        a_t(:,ele) = sqrt(1/N)*exp(sqrt(-1)*pi*sin(phi_t(ele))*(0:N-1));  %%% array response vector at transmitter
        temp_sum = temp_sum + alpha(ele)*a_r(:,ele)*a_t(:,ele)';
    end
    H(r,:) = sqrt(N*M/(1*los+0.3162*non_los))*temp_sum;
end
temp = sqrt(1/2)*(randn(R,N) + sqrt(-1)*randn(R,N));   %%% this is a random variable whose power is same as the power of H

%temp = 10*sqrt(1/2)*(randn(R,N) + sqrt(-1)*randn(R,N));   %%% this is a random variable whose power is 20 dB more than the power of H

%temp = (1/10)*sqrt(1/2)*(randn(R,N) + sqrt(-1)*randn(R,N));   %%% this is a random variable whose power is 20 dB less than the power of H

temp = reshape(temp,R,1,N);
H = reshape(H,R,1,N);
H_est = H + temp;

%val = H;
%save('C:/Users/Vaishnavi/Desktop/6 sem project (capstone)/2users-perfect/Test/H_2.mat','val')
%val = H_est;
%save('C:/Users/Vaishnavi/Desktop/6 sem project (capstone)/2users-perfect/Test/H_2_est.mat','val')

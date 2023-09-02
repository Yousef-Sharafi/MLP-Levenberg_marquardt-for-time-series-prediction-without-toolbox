%Programmer: Yousef Sharafi

clc;
close all;
clear all;

data=xlsread('Temperature Dataset.xlsx');
num_data=size(data,1);

for ii=1:4
    data(:,ii)=data(:,ii)/max(data(:,ii));
end

percent_train=0.75;
num_train=round(num_data*percent_train);
num_test=num_data-num_train;

n1=3;
n2=100;
n3=20;
n4=1;

eta=0.01;
epoch=40;

mse_train=zeros(1,epoch);
mse_test=zeros(1,epoch);

a=-1;
b=1;

w1=unifrnd(a,b,[n2 n1]);
net1=zeros(n2,1);
o1=zeros(n2,1);
pw1=zeros(n2,n1);

w2=unifrnd(a,b,[n3 n2]);
net2=zeros(n3,1);
o2=zeros(n3,1);
pw2=zeros(n3,n2);

w3=unifrnd(a,b,[n4 n3]);
net3=zeros(n4,1);
o3=zeros(n4,1);
pw3=zeros(n4,n3);

w_par=zeros(num_train,n2*n1+n3*n2+n4*n3,1);
I=eye(n2*n1+n3*n2+n4*n3);

for t=1:epoch
    error=zeros(1,num_train);
    for i=1:num_train
        input=data(i,1:3);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=(net3);
        target=data(i,4);
        error(i)=target-o3;
        
        t1=o2.*(1-o2);
        A=diag(t1);
        
        t2=o1.*(1-o1);
        B=diag(t2);
        %
        %             w1=w1-eta*error(i)*-1*(w3*A*w2*B)'*input;
        %             w2=w2-eta*error(i)*-1*(w3*A)'*o1';
        %             w3=w3-eta*error(i)*-1*1*o2';
        pw1=-1*(w3*A*w2*B)'*input;
        pw2=-1*(w3*A)'*o1';
        pw3=-1*1*o2';
        
        a=reshape(pw1,numel(pw1),1)';
        b=reshape(pw2,numel(pw2),1)';
        c=reshape(pw3,numel(pw3),1)';
        w_par(i,:)=[a b c];
    end
    
    a1=reshape(w1,numel(w1),1)';
    b1=reshape(w2,numel(w2),1)';
    c1=reshape(w3,numel(w3),1)';
    
    w_par1=[a1 b1 c1];
        
    miu = 1 * ( error * error');
    w_par1  = ( w_par1' - inv( w_par' * w_par + miu * I) * w_par' * error')';
    
    a2=w_par1(1:numel(w1));
    b2=w_par1(numel(w1)+1:numel(w1)+numel(w2));
    c2=w_par1(numel(w2)+numel(w1)+1:numel(w2)+numel(w1)+numel(w3));
    
    w1=reshape(a2,n2,n1);
    w2=reshape(b2,n3,n2);
    w3=reshape(c2,n4,n3);
    
    error_data_train=zeros(1,num_train);
    output_data_train=zeros(1,num_train);
    for i=1:num_train
        input=data(i,1:3);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=(net3);
        target=data(i,4);
        output_data_train(i)=o3;
        error_data_train(i)=target-o3;
    end
    mse_train(t)=mse(error_data_train);
    
    error_data_test=zeros(1,num_test);
    output_data_test=zeros(1,num_test);
    for i=1:num_test
        input=data(num_train+i,1:3);
        net1=w1*input';
        o1=logsig(net1);
        net2=w2*o1;
        o2=logsig(net2);
        net3=w3*o2;
        o3=(net3);
        target=data(num_train+i,4);
        output_data_test(i)=o3;
        error_data_test(i)=target-o3;
        
    end
    mse_test(t)=mse(error_data_test);
    
    figure(1);
    subplot(2,2,1),plot(data(1:num_train,4));
    hold on;
    plot(output_data_train,'r','linewidth',1);
    hold off;
    xlabel('Train Data');
    ylabel('Output');
    
    subplot(2,2,2),semilogy(mse_train(1:t));
    hold off;
    xlabel('Epoch');
    ylabel('mse train');
    
    subplot(2,2,3),plot(data(num_train+1:num_data,4));
    hold on;
    plot(output_data_test,'r','linewidth',1);
    hold off;
    xlabel('Test Data');
    ylabel('Output');
    
    subplot(2,2,4),semilogy(mse_test(1:t));
    hold off;
    xlabel('Epoch');
    ylabel('mse test');
end

figure(2);
plotregression(data(1:num_train,4),output_data_train);
title('Regression Train');

figure(3);
plotregression(data(num_train+1:num_data,4),output_data_test);
title('Regression Test');

mse_train_result = mse_train(epoch)
mse_test_result = mse_test(epoch)





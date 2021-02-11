clear all
sig=1e3;
num_eigs=2;
flag_weighted=0;

gnd = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\cifar_mlp_2layer\labels.csv');
[Gx,Ax]=mtx2graph('GfCIFARin_CLEVER.mtx');
[Gy,Ay]=mtx2graph('GfCIFAR_CLEVER_mlp.mtx');
[TopEig_cifar_mlp, TopEdgeList_cifar_mlp, TopNodeList_cifar_mlp]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
[Gy,Ay]=mtx2graph('GfCIFAR_CLEVER_cnn.mtx');
[TopEig_cifar_cnn, TopEdgeList_cifar_cnn, TopNodeList_cifar_cnn]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
[Gy,Ay]=mtx2graph('GfCIFAR_CLEVER_dd.mtx');
[TopEig_cifar_dd, TopEdgeList_cifar_dd, TopNodeList_cifar_dd]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;

%load labels
% load ../cifar10/output0;
% [Gx,Ax]=mtx2graph('GfCIFARin0.mtx');
% [Gy,Ay]=mtx2graph('GfCIFAR0.mtx');
% [dist0]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% % figure;plot(abs(dist0),'r-o');
% %  hold on
% 
%  load cifar10/output025;
% [Gx,Ax]=mtx2graph('GfCIFARin025.mtx');
% [Gy,Ay]=mtx2graph('GfCIFAR025.mtx');
% [dist025]=RiemannianDist(Gx,Gy,num_eigs,gnd);
% % plot(dist025,'b-*');
% 
% load cifar10/output05;
% [Gx,Ax]=mtx2graph('GfCIFARin05.mtx');
% [Gy,Ay]=mtx2graph('GfCIFAR05.mtx');
% [dist05]=RiemannianDist(Gx,Gy,num_eigs,gnd); 
% plot(dist05,'g->');

% load cifar10/output1;
% [Gx,Ax]=mtx2graph('GfCIFARin1.mtx');
% [Gy,Ay]=mtx2graph('GfCIFAR1.mtx');
% [dist1]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
%  plot(dist1,'k-<');
%  legend({'\epsilon=0','\epsilon=0.25','\epsilon=0.5','\epsilon=1.0'},'Location','southeast','Fontsize',18)
% legend({'\epsilon=0','\epsilon=0.5','\epsilon=1.0'},'Location','southeast','Fontsize',18)
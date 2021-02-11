clear all
num_eigs=2;
%load labels
% load MNIST/input0;
% gnd=double(gnd);
% gnd = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\mnist_mlp_2layer\mnist_mlp2layer_labels.csv');
gnd = readmatrix('I:\process\mltest\mnist_data_0\gbt_label_test.csv');

% %load graphs
% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST0self.mtx');
% [TopEig0_self, TopEdgeList0_self, TopNodeList0_self]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% 
% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST0dl.mtx');
% [TopEig0_dl, TopEdgeList0_dl, TopNodeList0_dl]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;


% figure;plot(dist0,'r-o');
% hold on
% gnd = readmatrix('I:\process\mltest\mnist_data_0.1\gbt_label_test.csv');
% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST01.mtx');
% [TopEig01, TopEdgeList01, TopNodeList01]=RiemannianDist(Gx,Gy,num_eigs,gnd);
% % plot(dist01,'b-*');

% gnd = readmatrix('I:\process\mltest\mnist_data_0.2\gbt_label_test.csv');
% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST02.mtx');
% [TopEig02, TopEdgeList02, TopNodeList02]=RiemannianDist(Gx,Gy,num_eigs,gnd);
% plot(dist02,'g->');

% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST03self.mtx');
% [TopEig03_self, TopEdgeList03_self, TopNodeList03_self]=RiemannianDist(Gx,Gy,num_eigs,gnd);
% 
% [Gx,Ax]=mtx2graph('GfMNISTin0.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST03dl.mtx');
% [TopEig03_dl, TopEdgeList03_dl, TopNodeList03_dl]=RiemannianDist(Gx,Gy,num_eigs,gnd);
% plot(dist03,'k-<');
%
[Gx,Ax]=mtx2graph('GfMNISTin_CLEVER.mtx');
[Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_CLEVER_mlp.mtx');
[TopEig_mnist_mlp, TopEdgeList_mnist_mlp, TopNodeList_mnist_mlp]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;

[Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_CLEVER_cnn.mtx');
[TopEig_mnist_cnn, TopEdgeList_mnist_cnn, TopNodeList_mnist_cnn]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;

[Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_CLEVER_dd.mtx');
[TopEig_mnist_dd, TopEdgeList_mnist_dd, TopNodeList_mnist_dd]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% legend({'\epsilon=0','\epsilon=0.1','\epsilon=0.2','\epsilon=0.3'},'Location','southeast','Fontsize',18)

% [Gx,Ax]=mtx2graph('GfMNISTin.mtx');
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_0.mtx');
% [TopEig_mnist_0, TopEdgeList_mnist_0, TopNodeList_mnist_0]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% 
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_01.mtx');
% [TopEig_mnist_01, TopEdgeList_mnist_01, TopNodeList_mnist_01]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% 
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_02.mtx');
% [TopEig_mnist_02, TopEdgeList_mnist_02, TopNodeList_mnist_02]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% 
% [Gy,Ay]=mtx2graph('RobustTraining/GfMNIST_03.mtx');
% [TopEig_mnist_03, TopEdgeList_mnist_03, TopNodeList_mnist_03]=RiemannianDist(Gx,Gy,num_eigs,gnd) ;
% [dist0]=RiemannianDist(Gy,Gy1,num_eigs,sig,flag_weighted) ;
% figure;plot(dist0,'r-o');
%This code implements GRASPEL algorithm for graph spectral learning
% clear all; 
% load USPS;%data matrix
% gnd=gnd-1;%labels
% fnameOUT='GfUSPS1.mtx';%

% fnameOUT='GfMNIST0self.mtx';%
% fea = readmatrix('I:\process\mltest\mnist_data_0\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0\gbt_label_test.csv');

% fnameOUT='GfMNIST0dl.mtx';%
% fea = readmatrix('I:\process\mltest\mnist_data_0_dl\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0_dl\gbt_label_test.csv');

% fnameOUT='GfMNISTin01.mtx';%
% fea = readmatrix('I:\process\mltest\mnist_data_0.1\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0.1\gbt_label_test.csv');

% fnameOUT='GfMNIST02.mtx'
% fea = readmatrix('I:\process\mltest\mnist_data_0.2\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0.2\gbt_label_test.csv');

% fnameOUT='GfMNIST03self.mtx'
% fea = readmatrix('I:\process\mltest\mnist_data_0.3\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0.3\gbt_label_test.csv');

% fnameOUT='GfMNIST03dl.mtx'
% fea = readmatrix('I:\process\mltest\mnist_data_0.3_dl\gbt_output_out_test.csv');
% gnd = readmatrix('I:\process\mltest\mnist_data_0.3_dl\gbt_label_test.csv');
% load fea_airfoil;%data matrix
% 
% gnd=fea(:,1)*0;%labels
k=20;% for setting up the kNN graph
% fnameOUT='GfCIFAR_CLEVER_dd.mtx';%
% fea = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\cifar_cnn_dd_teacher\cifar_cnn_dd_teacher_output.csv');
% gnd = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\cifar_mlp_2layer\cifar_mlp2layer_labels.csv');

% fnameOUT='GfCIFAR_CLEVER_cnn.mtx';%
% fea = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\cifar_cnn_7layer\cifar_cnn7layer_output.csv');
% gnd = readmatrix('D:\Process\mltest\topology_data_set\newdataset\adversarial_examples_ch3\clever\CLEVER-master\gbt_output\cifar_mlp_2layer\cifar_mlp2layer_labels.csv');

fnameOUT='GfMNIST_02.mtx';%
fea = readmatrix('I:\process\mltest\mnist_data_0.2\gbt_output_out_test.csv');
gnd = readmatrix('I:\process\mltest\mnist_data_0\gbt_label_test.csv');


% load MNIST/input0;
% fnameOUT='GfMNISTin0.mtx';%final (output) graph after GRASPEL iterations
% % % % %
%  load MNIST/output03;
%  fnameOUT='GfMNIST03.mtx';%final (output) graph after GRASPEL iterations

% % % %  
% load ../cifar10/input0
% fnameOUT='GfCIFARin0.mtx';%final (output) graph after GRASPEL iterations

% load ../cifar10/output0;
% fnameOUT='GfCIFAR0.mtx';%final (output) graph after GRASPEL iterations

% load MNIST/input0;
% fnameOUT='GfMNISTin0.mtx';%final (output) graph after GRASPEL iterations
% gnd=double(gnd);
% % 
% load MNIST/output03;
% fnameOUT='GfMNIST03.mtx';%final (output) graph after GRASPEL iterations
% gnd=double(gnd);
% 
% load MNISTnew/sm_output03;
% fnameOUT='GfMNIST03.mtx';%final (output) graph after GRASPEL iterations
% gnd=double(gnd);
% load pendigits;
% load pendigits_label;
% fea=feature;
% gnd=label-1;
% load fashionmnisttest;
% gnd=(fashionmnisttest(:,1));
% fea=(fashionmnisttest(:,2:785));

DoSpectralCluster=0;
Subset=[0:9];%for selecting a subset of the data
num_cluster=length(Subset);%for selecting a subset of the data and spectral clustering
DoGRASPEL=0;%choose 1 if using GRASPEL iterations; choose 0 if only using kNN graph as output
num_eigs=50;%for spectral clustering
% NskipRow=1;% for selecting a subset of the data: when NskipRow=2, it means only 50% data points selected 
% NskipCol=1;% for selecting a subset of the data features: when NskipCol=2, it means only 50% features selected 

TestResistance=0;% choose 1 to check if effective resistance distances agree with data distance
samplePerc=1e-1;%top (bottom) perc. sorted nodes as candidate edges
edge_iter=1e-4;%edge budget per iteration: 1% means 1% # of nodes
sig=1e3;%feature variance
tol=10; %embedding distortion tolerance should be greater than 1.0
maxIter=20;%max GRASPEL iterations
num_test=10;%number of kmean clustering runs

% % Pick up a subset from the original data set
% [idx,val]=find(gnd==[Subset]);
% fea=fea(idx,:);
% gnd=gnd(idx);
% 
% idxRow=1:NskipRow:length(fea(:,1));
% idxCol=1:NskipCol:length(fea(1,:));
% fea=fea(idxRow,idxCol);
% 
% % idx2=1:Nskip:length(fea(:,1));
% % fea=fea(idx2,:);
% gnd=gnd(idxRow);
 
%data size
[num_node,dim]=size(fea);

%feature preprocessing
 fea=(fea-mean(fea,2));
% for i=1:dim
%     fea(i,:)=fea(i,:)/norm(fea(i,:));
% end
 fea=fea/norm(fea)*sqrt(dim);

%create the initial graph based on kNN graph
tic
fname='kNN.mtx';% Initial kNN graph before GRASPEL iterations
[G,A0]=getMatrixKNN(fea,k,30,fname); %original kNN
toc
if DoGRASPEL==1
%GRASPEL iteration starts here
[G,variationRatio,cost, numIter] = FiedlerConstruct(fea,G, samplePerc,edge_iter,sig,maxIter,tol);
end
Gf=G;
wmax=max(Gf.Edges.Weight);
wmin=min(Gf.Edges.Weight);
% Gf.Edges.Weight=Gf.Edges.Weight./min(Gf.Edges.Weight);

%Final graph density check
num_edge=(length(Gf.Edges.Weight));
density=num_edge/num_node
[L,A]=getLaplacian(Gf,sig); 
 
if DoSpectralCluster==1
%     SpectralDrawingLabel(Gf,gnd,3) ;
% else
%spectral clustering & drawing
D=L+A;
[Us, vals] = eigs(L , num_eigs,'smallestabs');

%solution quality measures using NMI and ACC for spectral clustering
avgnmi=[];%NMI metric
avgacc=[];%ACC metric

    %Final result is computed by averaging multiple runs
    for i=1:num_test
    [idxSC,C]=kmeans(Us(:,2:num_cluster),num_cluster);
    avgnmi=[avgnmi;nmi(idxSC,gnd+1)];
    avgacc=[avgacc;accuracy(gnd+1,idxSC)];
    end
    nmiRes=mean(maxk(avgnmi,num_test))
    accRes=mean(maxk(avgacc,num_test))
    figure;plot(diag(vals),'*');

    % spectral graph drawing using labels (edges are not shown)
    figure;plot(Gf,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',gnd,'edgecolor','none');
%         figure;plot(Gf,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',idxSC,'edgecolor','none');

%     figure;plot(Gf,'XData',Us(:,2),'YData',Us(:,3),'NodeCData',gnd,'edgecolor','none');
%     figure;plot(log(variationRatio))
end

%test distance embedding
if TestResistance==1
   [Rdist,Xdist, EdgeList,r]=testResistance(Gf,fea, 1000,sig);
   figure;plot(Rdist,Xdist,'*');
end
% Gl=laplacian(Gf);
% D=diag(Gl);
% figure;hist(D)
%store the matrix for the learned graph
 mtxwrite(fnameOUT,L);

 sum(conncomp(Gf)~=1)

 
function [G,varRatio, costVec, numIter] = FiedlerConstruct(fea,G, samplePerc,chosenPercEachIter,sig,maxIter,tol)
varRatio=[];
costVec=[];
% tic
% %create the matrix based on kNN graph
% fname='kNN.mtx';% Initial kNN graph before GRASPEL iterations
% [G,A]=getMatrixKNN(fea,k,30,fname); %original kNN
% toc 

i=0;
tic

% Iteratively adding new edges and checking convergence
for i=1:maxIter 
    [G,var,cost] = FiedAddEdges(G,fea,samplePerc,chosenPercEachIter,sig,tol);
    if i==1
        cost0=cost;
    end
    varRatioPre=varRatio;
    varRatio=[varRatio;var];  
    costVec=[costVec;cost];
     if var <tol 
            break;
     end 
end

toc
numIter=i;

    
 

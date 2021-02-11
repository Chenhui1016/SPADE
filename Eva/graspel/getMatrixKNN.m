function [G0,A]=getMatrixKNN(fea,k,blockSize,fname)
if ( nargin <3) 
    blockSize=3;
end
[num_node,dim]=size(fea);
 
Graph0=gen_nn_distance(fea,k,blockSize,0);
G0 = graph(Graph0,'upper');
distX=   ((G0.Edges.Weight).^(2));
Weight=1./(distX); %weight=1/dist^2*dim
G0.Edges.Weight=1*Weight;% the max edge weight becomes 1

A=adjacency(G0,'weighted');
D=diag(sum(A,2));
L=D-A;

% spectral graph drawing
if 0
    [V,lamda]=eigs(L,4,'sm');
    figure('Name','2D spectral drawing of the sparsified Laplacian','NumberTitle','off');
    gplot(A,[V(:,2) V(:,3)],'b-');
end

if ( nargin == 4) 
    mtxwrite(fname,L);
end


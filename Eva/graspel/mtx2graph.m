function  [g,A]=mtx2graph(filename)
fileID = fopen(filename, 'r');
B = textscan(fileID, '%d %d %f64 %f64', 'headerlines', 1);
aa=cell2mat(B(1));
bb=cell2mat(B(2));
cc=cell2mat(B(3));
dd=cell2mat(B(4));
factor=1;
if ~(abs(cc(1))>=0)
    cc=ones(length(aa),1);
end
if dd(1)>0
  factor=dd(1);
end
dim=max(max(aa),max(bb));
A=sparse(double(aa), double(bb), cc/factor,double(dim),double(dim));
A=diag(sum(A))-A;
g=graph(A,'lower','omitselfloops');
g.Edges.Weight=abs(g.Edges.Weight);
% g.Edges.Weight=abs(rand(length(g.Edges.Weight),1))*10+1;

A=adjacency(g,'weighted');

% spectral graph drawing for visulization
if 0
    sig=1e3;
    D=diag(sum(A,2));
    L=D-A;
    L=L+1/sig^2*speye(size(L,1));
    num_cluster=10;
    [Us, vals] = eigs(L,num_cluster,'smallestabs');
    [idx,C]=kmeans(Us(:,1:num_cluster),num_cluster);
    figure('Name','2D spectral graph drawing w/ spectral clustering','NumberTitle','off');
%     plot(g,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',idx,'edgecolor','none');
    plot(g,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',idx);
    figure;plot(diag(vals),'*');
end


function  [g,A,weightMax]=mtx2feagraph(fea,filename)
fileID = fopen(filename, 'r');
B = textscan(fileID, '%d %d %f64 %f64', 'headerlines', 1);
aa=cell2mat(B(1));
bb=cell2mat(B(2));
cc=cell2mat(B(3));
dd=cell2mat(B(4));
factor=1;
if dd(1)>0
  factor=dd(1);
end
A = sparse(double(aa), double(bb), cc/factor);
A=diag(sum(A))-A;
g=graph(A,'lower','omitselfloops');
g.Edges.Weight=abs(g.Edges.Weight);
weightMax=median(g.Edges.Weight);
g.Edges.Weight=g.Edges.Weight/weightMax;% the max edge weight becomes 1
p=g.Edges.EndNodes(1,1);
q=g.Edges.EndNodes(1,2);
weight=g.Edges.Weight(1); 
weight0=1/getDistance(fea,p,q);
weightMax=weight0/weight;
A=adjacency(g,'weighted');

% spectral graph drawing for visulization
if 0
    D=diag(sum(A,2));
    L=D-A;
    num_cluster=10;
    [Us, vals] = eigs(L,num_cluster,'smallestabs');
    [idx,C]=kmeans(Us(:,1:num_cluster),num_cluster);
    figure('Name','3D spectral graph drawing w/ spectral clustering','NumberTitle','off');
    plot(g,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',idx,'edgecolor','none');

end


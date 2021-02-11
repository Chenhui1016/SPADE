function [Rdist]=getResistance(graph,sig,EdgeList,flag_weighted)
num_edge=length(EdgeList(:,1));
if flag_weighted
     [L,A]=getLaplacian(graph,sig); 
else
    [L]=laplacian(graph); 
end
L=L+1/sig^2*speye(size(L,1)); 
num_node=length(L(:,1));

Rdist=[];% Eff Res. distance
[R, pp, S] = chol(L);

for i=1:num_edge
    p=EdgeList(i,1);
    q=EdgeList(i,2);
    epq=zeros(num_node,1);
    epq(p)=1;
    epq(q)=-1;
    x= S*(R\(R'\(S'*epq)));
    Rpq=epq'*x;
    Rdist=[Rdist;Rpq];   
end


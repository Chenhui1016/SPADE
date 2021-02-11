function [Rdist,Xdist, EdgeList,r]=testResistance(graph,fea,num_sample,sig)
[num_node,dim]=size(fea);
samplePerc=0.05;
num_eigs=2;
A=adjacency(graph,'weighted');
D=diag(sum(A,2));
L=D-A; 
L=L+1/sig^2*speye(size(L,1)); 
[Us, vals] = eigs(L ,num_eigs,'smallestabs','Tolerance',1e-6); 
u2=Us(:,2);

 %Graph node sorting
[Y,I]=sort(u2);
topPart=I(1:floor(num_node*samplePerc));
botPart=I(num_node-floor(num_node*samplePerc)+1:num_node);


Rdist=[];% Eff Res. distance
Xdist=[];% Data distance
EdgeList=[];
[R, pp, S] = chol(L);

for i=1:num_sample
    rng(i);
    a=randi(length(topPart)); 
    b=randi(length(botPart)); 
    a_idx=topPart(a);
    b_idx=botPart(b);
    
    %always set p to have the bigger node index 
    p=max([a_idx b_idx]);
    q=min([a_idx b_idx]);
    epq=zeros(num_node,1);
    epq(p)=1;
    epq(q)=-1;
    x= S*(R\(R'\(S'*epq)));
    Rpq=epq'*x;
    Xpq=getDistance(fea,p,q);
    Rdist=[Rdist;Rpq];
    Xdist=[Xdist;Xpq];
    EdgeList=[EdgeList;p q];
end
% figure
% plot(Rdist,Xdist,"*")
r=corrcoef(Rdist,Xdist)

function [L,A]=getLaplacian(Graph,sig)
    A=adjacency(Graph,'weighted');
    D=diag(sum(A,2));
    L=D-A;
    L=L+1/sig^2*speye(size(L,1));  
    
 

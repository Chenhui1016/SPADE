function [Rdist]=getGeoDist(graph,EdgeList,flag_weighted)
num_edge=length(EdgeList(:,1));
Rdist=[];% Eff Res. distance

for i=1:num_edge
    p=EdgeList(i,1);
    q=EdgeList(i,2);
    if flag_weighted
        [path Rpq]=shortestpath(graph,p,q);
    else
        [path Rpq]=shortestpath(graph,p,q,'Method','unweighted');
    end
    Rdist=[Rdist;Rpq];   
end


function [TopEig, TopEdgeList, TopNodeList]=RiemannianDist(Gx,Gy,num_eigs,gnd) 
dist=[];
[Lx]=laplacian(Gx); %input graph
[Ly]=laplacian(Gy); %output graph
[Uxy, Dxy]=eigs(Lx,Ly,num_eigs);%generalized eigenvalue
num_node_tot=length(Uxy(:,1));
num_top_node=10;
dist=zeros(1,1);
num_cluster=5;
TopEig=max(diag(Dxy));
NodeDegree=diag(Lx);
%find the top edges most important to the largest eigenvalue
if 1    
    num_edge_tot=length(Gx.Edges.Weight); % number of total edges    
    Zpq=zeros(num_edge_tot,1);% edge embedding distance
    p=Gx.Edges.EndNodes(:,1);% one end node of each edge
    q=Gx.Edges.EndNodes(:,2);% another end node of each edge
    density=ceil(num_edge_tot/num_node_tot);
    %Compute spectral embedding distance for each edge in Gx
    for i=1:num_eigs
        Zpq=Zpq+(Uxy(p,i)-Uxy(q,i)).^2*Dxy(i,i);
    end    
    Zpq=Zpq./max(Zpq); 
    
    %compute node scores
    node_score=zeros(num_node_tot,1);
    label_score=zeros(length(unique(gnd)),1);
    for i=1:num_edge_tot
        node_score(p(i))=node_score(p(i))+Zpq(i);
        node_score(q(i))=node_score(q(i))+Zpq(i);
        
    end 
    node_score=node_score./NodeDegree;
    node_score=node_score./max(node_score); 
%     node_score=exp(node_score*3);
    
    %compute label scores
    for i=1:num_node_tot
        idx=gnd(i);
        label_score(idx+1)=label_score(idx+1)+node_score(i);        
    end
    
    label_score=label_score./max(label_score);
%    
    [YY,II]=sort(node_score,'descend');
    TopNodeList=II;
    node_top=II(1:num_top_node);         
     gndSort=gnd(II);
    %edges with top embedding distances
    [Y,I]=sort(Zpq,'descend');
    TopEdgeList=Gx.Edges.EndNodes(I,:);
    
if 0
    if 0
        num_top_edge=length(find(Y(:,1)>tol))
        num_top_node=ceil(num_top_edge/density);% number of selected nodes
    else
         num_top_edge=1000;
         num_top_node=ceil(num_top_edge/density);% number of selected nodes
    end
    pp=p(I(1:num_top_edge));
    qq=q(I(1:num_top_edge));
    NodeList= [pp ; qq]; %store all end nodes into one vector
    Rdist=[];% store shortest path distances
    LabelDiffList=[];% store label difference of each edge's end nodes
    
    for i=1:num_top_edge
        idxMax=I(i);
%           idxMax=randi(num_edge_tot);
        pMax=p(idxMax);
        qMax=q(idxMax); 
        NodeList=[NodeList;pMax;qMax];
        [path Rpq]=shortestpath(Gy,pMax,qMax,'Method','unweighted');
        LabelDiffList=[LabelDiffList; length(unique(gnd(path)))];
        [gnd(path)];
        Rdist=[Rdist; Rpq];
    end
    
    topR=maxk(Rdist, num_top_edge);
    [sum(topR)/num_top_edge sum(LabelDiffList)/num_top_edge];
    
    %find top few critical (most frequent) data points (end nodes)    
    tempNode = unique(NodeList);
    outNode = [tempNode,histc(NodeList(:),tempNode)];
    [B,I]=sort(outNode(:,2),'descend');
    topNodeList=outNode(I(1:num_top_node),:);
    topLabelList=gnd(topNodeList(:,1));
    tempLabel = unique(topLabelList);
    outLabel = [tempLabel,histc(topLabelList(:),tempLabel)];
end

    %Draw the graph
   if 0
%         SpectralDrawingLabel(Gx,gnd,2) ;
%        SpectralDrawingLabel(Gy,gnd,2) ;

        [idxC,C]=kmeans(Uxy(:,1:num_eigs),num_cluster);
%           figure('Name','2D spectral graph drawing','NumberTitle','off');
%            plot(Gx,'XData',Uxy(:,1),'YData',Uxy(:,2),'NodeCData',gnd,'EdgeColor','none');
%              figure; bar(0:9,label_score);
%           plot(Gy,'XData',Uxy(:,1),'YData',Uxy(:,2),'ZData',Uxy(:,3),'NodeCData',node_score,'EdgeColor','none');

%          plot(Gy,'XData',Uxy(:,1),'YData',Uxy(:,2),'ZData',Uxy(:,3),'NodeCData',node_score,'EdgeColor','none');
        distC=zeros(num_cluster,1);
        for ii=1:num_cluster
            for jj=1:num_cluster
                distC(ii)=distC(ii)+norm(C(ii,:)-C(jj,:));
            end
        end
        [maxDist,maxIdx]=max(distC);
        ratio=[];
        for ii=1:num_cluster
            x=zeros(num_node_tot,1);
            x(find(idxC==ii))=1;
            ratio=[ratio (x'*Lx*x)/(x'*Ly*x)];
        end
        ratio;
        
   end
   
end
% 
% for i=1:1
%     for j=1:i
%         dist(i)=dist(i)+(log(Dxy(j,j)))^2+(log(Dxy(j,j)))^2;
%     end
%     dist(i)=sqrt(dist(i));
% end
 
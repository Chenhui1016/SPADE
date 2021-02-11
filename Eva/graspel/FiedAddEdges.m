function [G,variationRatio,cost]=FiedAddEdges(G,fea,samplePerc,chosenPercEachIter,sig,tol)
num_eigs=5;%number of eigenvectors for spectral graph embedding
[n,dim]=size(fea);
[L,A]=getLaplacian(G,sig); 
[Us, vals] = eigs(L ,num_eigs,'smallestabs','Tolerance',1e-6); 
 
%Adjust weights for each eigenvector
for i=2:num_eigs
    Us(:,i)=Us(:,i)/sqrt(vals(i,i));
end

%Fiedler vector
u2=Us(:,2);

%Graph node sorting
[Y,I]=sort(u2);
topPart=I(1:floor(n*samplePerc));
botPart=I(n-floor(n*samplePerc)+1:n);

chosenSet=zeros(ceil(n*chosenPercEachIter),5); 
[min_delta,location] = min(chosenSet(:,3));
 
for i=1:size(chosenSet,1)*1000 %sampling candidate edges
%         rng(i);
        a=randi(length(topPart)); 
        b=randi(length(botPart)); 
        a_idx=topPart(a);
        b_idx=botPart(b);
        %always set p to have the bigger node index 
        p=max([a_idx b_idx]);
        q=min([a_idx b_idx]);
        e_pq=zeros(n,1);
        e_pq(p)=1;
        e_pq(q)=-1;
        distX=getDistance(fea,p,q);        
        weight=1/(distX);
        distZ=getDistance(Us,p,q)*dim;
        delta_fiedler=distZ*weight;%spectral embedding distortion
        p_vec=chosenSet(:,1);
        q_vec=chosenSet(:,2);
        %check if the edge has been selected before
        flagP= nnz(find(p_vec==p));
        flagQ= nnz(find(q_vec==q));
        if delta_fiedler>min_delta && (flagP<1 || flagQ<1) && delta_fiedler>tol
            chosenSet(location,1)=p;
            chosenSet(location,2)=q;
            chosenSet(location,3)=delta_fiedler;
            chosenSet(location,4)=weight*1;% scale weight up by 10
%             chosenSet(location,4)=1;% scale weight up by 10
            chosenSet(location,5)=(1-1/delta_fiedler)*distZ; %gradient
            [min_delta,location] = min(chosenSet(:,3));
        end
end

%glasso cost function
if 0
    lambda=diag(vals);
    cost=log(prod(lambda(2:num_eigs)))-trace(fea'*L*fea')/dim;
else
    cost= max(chosenSet(:,5));
end

variationRatio=(mean(chosenSet(:,3))) %average embedding distortion

% convergence criteria
if min(chosenSet(:,3))==0 
    return; 
end 

%adding the selected edges into the latest graph
G=addedge(G,chosenSet(:,1),chosenSet(:,2),chosenSet(:,4));
G = simplify(G);
 




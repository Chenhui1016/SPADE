function [eigVal, eigVec, A, L]=SpectralDrawing(graph,dim,sig,num_eigs) 
    [L,A]=getLaplacian(graph,sig);
    
     D=L-A;
    [Us, vals] = eigs(L, num_eigs,'smallestabs');
    [idx,C]=kmeans(Us(:,2:num_eigs),num_eigs);
    eigVal=diag(vals);
    eigVec=Us;
    if dim==2
        figure('Name','2D spectral graph drawing w/ spectral clustering','NumberTitle','off');
        plot(graph,'XData',Us(:,2),'YData',Us(:,3),'NodeCData',idx);
    elseif dim ==3
        figure('Name','3D spectral graph drawing w/ spectral clustering','NumberTitle','off');
        plot(graph,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',idx);
    else 
          return;  
    end
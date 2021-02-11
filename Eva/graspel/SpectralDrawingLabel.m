function [eigVal, eigVec, L]=SpectralDrawingLabel(graph,label,dim) 
    label=double(label);
     num_eigs=length(unique(label));
     [L]=laplacian(graph);
    [Us, vals] = eigs(L, num_eigs,'smallestabs');
    [idx,C]=kmeans(Us(:,2:num_eigs),num_eigs);
    eigVal=diag(vals);
    eigVec=Us;
    if dim==2
        figure('Name','2D spectral graph drawing w/ spectral clustering','NumberTitle','off');
        plot(graph,'XData',Us(:,2),'YData',Us(:,3),'NodeCData',label,'edgecolor','none');
    elseif dim ==3
        figure('Name','3D spectral graph drawing w/ spectral clustering','NumberTitle','off');
        plot(graph,'XData',Us(:,2),'YData',Us(:,3),'ZData',Us(:,4),'NodeCData',label,'edgecolor','none');
    else 
          return;  
    end
N = 1000

% topnode_mnist_mlp = TopNodeList_mnist_mlp(1:N)-1
% csvwrite('ids_mnist_mlp.csv', topnode_mnist_mlp);
% topnode_mnist_cnn = TopNodeList_mnist_cnn(1:N)-1
% csvwrite('ids_mnist_cnn.csv', topnode_mnist_cnn);
% topnode_mnist_dd = TopNodeList_mnist_dd(1:N)-1
% csvwrite('ids_mnist_dd.csv', topnode_mnist_dd);
% 
% topnode_cifar_mlp = TopNodeList_cifar_mlp(1:N)-1
% csvwrite('ids_cifar_mlp.csv', topnode_mnist_mlp);
% topnode_cifar_cnn = TopNodeList_cifar_cnn(1:N)-1
% csvwrite('ids_cifar_cnn.csv', topnode_mnist_cnn);
% topnode_cifar_dd = TopNodeList_cifar_dd(1:N)-1
% csvwrite('ids_cifar_dd.csv', topnode_mnist_dd);

% topnode_mnist_0 = TopNodeList_mnist_0(1:N)-1
% csvwrite('ids_mnist_0.csv', topnode_mnist_0);
% topnode_mnist_01 = TopNodeList_mnist_01(1:N)-1
% csvwrite('ids_mnist_01.csv', topnode_mnist_01);
% topnode_mnist_02 = TopNodeList_mnist_02(1:N)-1
% csvwrite('ids_mnist_02.csv', topnode_mnist_02);
% topnode_mnist_03 = TopNodeList_mnist_03(1:N)-1
% csvwrite('ids_mnist_03.csv', topnode_mnist_03);


topedge_mnist_mlp = TopEdgeList_mnist_mlp(1:N, :)-1
csvwrite('edges_mnist_mlp.csv', topedge_mnist_mlp)
topedge_mnist_cnn = TopEdgeList_mnist_cnn(1:N, :)-1
csvwrite('edges_mnist_cnn.csv', topedge_mnist_cnn)
topedge_mnist_dd = TopEdgeList_mnist_dd(1:N, :)-1
csvwrite('edges_mnist_dd.csv', topedge_mnist_dd)

topedge_cifar_mlp = TopEdgeList_cifar_mlp(1:N, :)-1
csvwrite('edges_cifar_mlp.csv', topedge_cifar_mlp)
topedge_cifar_cnn = TopEdgeList_cifar_cnn(1:N, :)-1
csvwrite('edges_cifar_cnn.csv', topedge_cifar_cnn)
topedge_cifar_dd = TopEdgeList_cifar_dd(1:N, :)-1
csvwrite('edges_cifar_dd.csv', topedge_cifar_dd)

topedge_mnist_0 = TopEdgeList_mnist_0(1:N, :)-1
csvwrite('edges_mnist_0.csv', topedge_mnist_0)
topedge_mnist_01 = TopEdgeList_mnist_01(1:N, :)-1
csvwrite('edges_mnist_01.csv', topedge_mnist_01)
topedge_mnist_02 = TopEdgeList_mnist_02(1:N, :)-1
csvwrite('edges_mnist_02.csv', topedge_mnist_02)
topedge_mnist_03 = TopEdgeList_mnist_03(1:N, :)-1
csvwrite('edges_mnist_03.csv', topedge_mnist_03)
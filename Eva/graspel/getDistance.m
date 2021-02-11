function [dist]= getDistance(fea,p,q)
dist=   ((norm(fea(p,:)-fea(q,:)))^(2));
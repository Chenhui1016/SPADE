function [xx, t]=ElementFreq(x) 
xx = unique(x);       % temp vector of vals
x = sort(x);          % sorted input aligns with temp (lowest to highest)
t = zeros(size(xx)); % vector for freqs
% frequency for each value
for i = 1:length(xx)
    t(i) = sum(x == xx(i));
end
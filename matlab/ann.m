function [ neighbors ] = ann( data, index, obj, k )
%Finds the 'k' approximate nearest neighbors of 'obj' using the mrpt index
%'index'
%   Detailed explanation goes here
neighborhood = [];
for t = 1:size(index,2)
    neighborhood = union(neighborhood, find_leaf(index{t}, obj));
end
[dist, ind] = sort(pdist2(obj, data(neighborhood,:)));
neighbors = neighborhood(ind(1:k))';

% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ neighbors ] = ann( data, index, obj, k )
%Finds the 'k' approximate nearest neighbors of 'obj' using the mrpt index
%'index'
%
%   INPUTS:
%   data - The same data, unmodified, for which the index was built
%   index - A pre-built mrpt index
%   obj - The data object whose neighbors are beings searched for
%   k - The desired number of nearest neighbors
%
%   OUTPUTS:
%   neighbors - The indexes of the approximate neighbors in the data

% Find the set of potential neighbors by querying each of the trees in the
% index.
neighborhood = [];
for t = 1:size(index,1)
    neighborhood = union(neighborhood, find_leaf(index{t}, obj));
end

% Find the k nearest objects within the potential neighbors.
[~, ind] = sort(pdist2(obj, data(neighborhood,:)));
neighbors = neighborhood(ind(1:k))';

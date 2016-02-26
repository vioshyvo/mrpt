% Finds the approximate nearest neighbours of the query point using the
% pre-built tree structures.

% INPUTS:
% query - The query point as a vector
% trees - the trees as an array of structs as built by the
% build_rp_tree_recursive -function. The trees only store the points by
% their indices to save space, so the data has to be given separately.
% data - the data to find the vectors corresponding to the indices in the
% tree structures.
% k - The number of nearest neighbours wished to be found. Note that
% depending on the trees, less than k neighbours are sometimes returned. 

% OUTPUTS:
% neighbours - The indices of the k (or less) approximate nearest 
% neighbours in the data given as input, orderded by the distance to the 
% query point.
% The distances between the query point and the neighbours returned ordered.

function [neighbors, distances,S] = approximate_kNN(query, trees, data, k)

% Multiple rp-tree query returns the points that belong to the same leaf
% node in any of the trees.
neighbors = multiple_rp_tree_query(trees, query);
S=size(neighbors,2);

% Next, the linear search is performed in this search space
distances = pdist2(query, data(neighbors,:));
[distances, indices] = sort(distances);
neighbors = neighbors(indices);

% If the search space yielded by the trees is smaller than k, return all
% the points in the space ordered. Otherwise return only the k nearest
% neighbours in the search space.
if(k<size(neighbors,2))
   neighbors = neighbors(1:k);
   distances = distances(1:k);
end

end
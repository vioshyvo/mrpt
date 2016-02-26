% The function builds a random projection tree of the data. Leaf size is
% used as the stopping criterion.

% INPUTS:
% data - The data to be indexed. One data object per row, one variable per
% column.
% indices - The indices of the data to be indexed. Normally 1:size(data,2).
% n0 - The maximum number of data objects in a cluster / rp-tree leaf.
% splitFunction - The function used to split the 1-dimensional projections.
% Must take an 1-d array as input and output one real value, the split
% point.

% OUTPUTS:
% A struct that represents the root of the rp-tree. The struct has 4
% fields:
% left - A struct that represents the root node of the left subtree.
% right - A struct that represents the root node of the right subtree.
% splitPoint - The value that was used as the splitting criterion. The data
% points whose projection value is less than or equal to splitPoint were 
% assigned to the left subtree, and the rest to the right subtree.
% seed - The random seed used to generate the random vector.

% The subtree nodes have an identical structure with the exception of leaf
% nodes (clusters) which only have two fields, 'seed' and 'indices'. A seed
% value of '-1' indicates a leaf node and indices contains the indices of
% the data objects that belong to the cluster.

function [tree] =  build_rp_tree_recursive(data, indices, n0, splitFunction)

dim = size(data,2); % Dimensionality
N = size(indices,2); % The number of data objects at this node

if (N>n0)
    
    % Set the random seed
    randomSeed = randi(1.0e9);
    rng(randomSeed);
    
    % Select random unit vector
    randomVector = normrnd(0,1,[1,dim]);
   
    % Take the projection on the chosen vector
    projectedData = randomVector*data(indices,:)';
    
    % Sort the projections and find the splitting point
    [projectionsOrdered, orderingOfIndices] = sort(projectedData);
    indices = indices(orderingOfIndices);
    [splitIndex, splitPoint] = splitFunction(projectionsOrdered);
    
    % Recursively generate the subtrees
    left = build_rp_tree_recursive(data, indices(1:splitIndex-1),n0, splitFunction);
    right = build_rp_tree_recursive(data, indices(splitIndex:N),n0, splitFunction);
    
    % Generate the node
    tree = struct('left',left, 'right',right, 'splitPoint', splitPoint, 'seed', randomSeed);
else
    % The node is of correct size and is not split any further. Set seed 
    % value to -1 to indicate that this is a leaf.
    tree = struct('seed',-1, 'indices', indices);
end
end

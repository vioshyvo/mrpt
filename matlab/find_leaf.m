% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ indexes ] = find_leaf( tree, query )
%Places the queried data point to a leaf and returns the corresponding
%indexes.
%   INPUTS:
%   tree - a random projection tree instance created by the function rptree
%   query - the data object that needs to be placed to a correct leaf
%
%   OUTPUTS:
%   indexes - The indexes of the data points in the leaf where the query
%   object belongs.

% Restore rng settings to ensure that the random matrix generated is the
% same as while building the tree.
rng(tree.seed);

% Compute the projections.
projections = query * normrnd(0, 1, tree.dim, tree.tree_depth);

% Move down the tree with respect to the projections and splits.
node = tree.root;
ii = 1;
while size(node.get_left(), 1) ~= 0
    if projections(ii) < node.get_split()
        node = node.get_left();
    else
        node = node.get_right();
    end
    ii = ii+1;
end

% Return the indexes of the data objects in the leaf.
indexes = node.get_indexes()';
end

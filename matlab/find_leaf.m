function [ indexes ] = find_leaf( tree, query )
%Places the queried data point to a leaf and returns the corresponding
%indexes
%   Detailed explanation goes here

% Restore rng settings
rng(tree.seed);

projections = query * normrnd(0, 1, tree.dim, tree.tree_depth);

% Move down the tree with respect to the projections and splits
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

% Return the indexes of the data objects in the leaf
indexes = node.get_indexes()';
end

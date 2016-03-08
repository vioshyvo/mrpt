% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ tree ] = rptree( data, n0 )
%The function iteratively builds a random projection tree data structure 
%for the given data.
%   INPUTS:
%   data - The data for which th user wants to build the index
%   n0 - The maximum number of elements in a leaf of the tree
%
%   OUTPUTS:
%   tree - The random projection tree as a cell array with 4 fields:
%   {1} root - The root node of the tree which also provides access to the
%   whole tree structure. The trees are defined by nodes which are
%   instances of the java class RPTNode.
%   {2} seed - The random seed used to build the tree.
%   {3} dim - Dimensionality of the data set.
%   {4} tree_depth - The maximum depth of the tree.


% Generate and store a random seed for rng
rng('shuffle');
seed = rand*1e9;
rng(seed);

% Java LinkedList used as a queue
import java.util.LinkedList;

% Compute some basic characteristics of the tree.
dim = size(data, 2);
n_random_vectors = ceil(log2(size(data,1)/n0))

% Compute the projections.
all_projections = data * normrnd(0, 1, dim, n_random_vectors);

% The main while-loop that builds the tree one level at a time.
root = RPTNode(1:size(data,1));
queue = LinkedList();
queue.add(root);
curr_level = 1;
curr_level_size = 1;
curr_level_occupancy = 0;
while queue.size() > 0
    
    % Pop next node to be handled.
    node = queue.remove();
    
    % Find out the indices in this node and the ordering of their
    % projections.
    indexes = node.get_indexes();
    node_size = size(indexes, 1);
    [sorted_projs, order] = sort(all_projections(indexes, curr_level));
    
    % Create children, add to queue if necessary.
    % In case the split is not equal the extra index goes to the left
    % branch
    split = (sorted_projs(floor(node_size/2 + 1))+sorted_projs(ceil(node_size/2)))/2;
    left = RPTNode(indexes(order(1:ceil(node_size/2))));
    right = RPTNode(indexes(order(ceil(node_size/2)+1:node_size)));
    node.set_children(left, right, split);
    if node_size > 2*n0+1
        queue.add(left);
        queue.add(right);
    elseif node_size == 2*n0 + 1
        queue.add(left);
    end
    
    % Keep track of the tree level.
    curr_level_occupancy = curr_level_occupancy + 1;
    if curr_level_size == curr_level_occupancy
       curr_level_size = 2*curr_level_size;
       curr_level_occupancy = 0;
       curr_level = curr_level + 1;
    end
end

% Create the struct as specified in the OUTPUTS section.
tree = struct('root', root, 'seed', seed, 'dim', dim, 'n_rvs', n_random_vectors);
end
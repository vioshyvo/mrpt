% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ tree ] = rptree( data, n0, seed )
%The function iteratively builds a random projection tree data structure 
%for the given data.
%   INPUTS:
%   data - The data for which th user wants to build the index
%   n0 - The maximum number of elements in a leaf of the tree
%   seed - Optional: A user-specified seed for controlling the tree
%   building
%
%   OUTPUTS:
%   tree - The random projection tree as a cell array with 4 fields:
%   {1} root - The root node of the tree which also provides access to the
%   whole tree structure. The trees are defined by nodes which are
%   instances of the java class RPTNode.
%   {2} seed - The random seed used to build the tree.
%   {3} dim - Dimensionality of the data set.
%   {4} tree_depth - The maximum depth of the tree.

% Use a random seed only if the user does not specify one.
if nargin < 3
   rng('shuffle'); % Ensure randomness (after queries)
   seed = rand*1e9;
end

% Java LinkedList used as a queue
import java.util.LinkedList;

% Set random seed to ensure that the same random vectors can be generated
% when querying the tree.
rng(seed);

% Compute some basic characteristics of the tree.
dim = size(data, 2);
tree_depth = ceil(log2(size(data,1)/n0) + 1); % TSEK TSEK

% Compute the projections.
all_projections = data * normrnd(0, 1, dim, tree_depth);

% The main while-loop that builds the tree one level at a time.
root = RPTNode(1:size(data,1));
queue = LinkedList();
queue.add(root);
curr_level = 2;
curr_level_size = 2;
curr_level_occupancy = 0;
while queue.size() > 0
    
    % Pop next node to be handled.
    node = queue.remove();
    
    % Find out the indices in this node and the ordering of their
    % projections.
    indexes = node.get_indexes();
    [sorted_projs, order] = sort(all_projections(indexes, curr_level));
    
    % Create children, add to queue if necessary.
    split = median(all_projections(indexes, curr_level)); % MAKE FASTER!
    left = RPTNode(indexes(order(1:floor(size(order,1)/2))));
    right = RPTNode(indexes(order(floor(size(order,1)/2)+1: size(order,1))));
    node.set_children(left, right, split);
    if size(indexes, 1)/2 > n0
        queue.add(left);
        queue.add(right);
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
tree = struct('root', root, 'seed', seed, 'dim', dim, 'tree_depth', tree_depth);
end
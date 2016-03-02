% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ tree ] = rptree( data, n0, seed )
%The function builds and returns a rp-tree
%   The function iteratively builds a random projection tree data structure
%   for the given data. NOTICE THAT THE FUNCTION IS DEPENDENT ON THE JAVA
%   CLASSES!

% Use a random seed only if the user does not specify one.
if nargin < 3
   seed = rand; 
end

% Java LinkedList used as a queue
import java.util.LinkedList;

% Restoring rng settings 
rng(seed);

% Something something
dim = size(data, 2);
tree_depth = ceil(log2(size(data,1)/n0) + 1); % TSEK TSEK
all_projections = data * normrnd(1, 0, dim, tree_depth);

% The main while-loop that builds the tree one level at a time.
root = RPTNode(1:size(data,1));
queue = LinkedList();
queue.add(root);
curr_level = 2;
curr_level_size = 2;
curr_level_occupancy = 0;
while queue.size() > 0
    
    % Pop next node to be handled
    node = queue.remove();
    
    % Find out the indices in this node and the ordering of their
    % projections
    indexes = node.get_indexes();
    [sorted_projs, order] = sort(all_projections(indexes, curr_level));
    
    % Create children, queue if necessary
    split = median(all_projections(indexes, curr_level)); % MAKE FASTER!
    left = RPTNode(indexes(order(1:floor(size(order,1)/2))));
    right = RPTNode(indexes(order(floor(size(order,1)/2)+1: size(order,1))));
    node.set_children(left, right, split);
    if size(indexes, 1)/2 > n0
        queue.add(left);
        queue.add(right);
    end
    
    % Keep track of the tree level
    curr_level_occupancy = curr_level_occupancy + 1;
    if curr_level_size == curr_level_occupancy
       curr_level_size = 2*curr_level_size;
       curr_level_occupancy = 0;
       curr_level = curr_level + 1;
    end
end
tree = struct('root', root, 'seed', seed, 'dim', dim, 'tree_depth', tree_depth);
end
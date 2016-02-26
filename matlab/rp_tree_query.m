% Find the correct leaf of the query point in the rp-tree and return the
% indices of the other points in the leaf. The leaf is found using the
% information on the splits saved in the tree structure, ie. fields 'seed'
% and 'splitPoint'.
%
% INUPTS:
% tree - a struct that represents the root node of the rp-tree (as given by 
% the 'build_rp_tree_recursive' function.
% queryPoint - The query point as a row vector
%
% OUTPUTS:
% searchSpace - A 1-d array (row) containing the indices of the data objects
% that belong to the same leaf with the query point.

function [ searchSpace ] = rp_tree_query( tree, queryPoint )
dim = size(queryPoint, 2); % Dimensionality

while(tree.seed~=-1)
   % Restore random number generator settings
   rng(tree.seed)
   
   % Regenerate the random vector used for projections
   randomVector = normrnd(0,1,[1,dim]);
   
   % Compute the projection
   projection = randomVector*queryPoint';
   
   % Decide into which subtree the query point belongs
   if(projection < tree.splitPoint)
       tree = tree.left;
   else
       tree = tree.right;
   end
end
% Return the indices of the objects in the leaf
searchSpace =  tree.indices;
end


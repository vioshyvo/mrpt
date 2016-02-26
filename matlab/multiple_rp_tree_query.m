% The function performs a query in each of the rp trees and returns the
% union of the results of each separate query.

% INPUTS:
% trees - an array of structs, each of which represents the root node of a
% rp-tree as given by build_rp_tree_recursive.
% queryPoint - a valid query data vector

% OUTPUTS:
% searchSpace - The indices of the members of the union

function [searchSpace] = multiple_rp_tree_query(trees, querypoint)

% Call rp_tree_query once for each tree and add the results to the
% searchSpace.
searchSpace=[];
for ii=1:size(trees,2)
    searchSpace = union(searchSpace,rp_tree_query(trees(ii),querypoint));
end

% Check that the result is a row vector, transpose if not.
if (size(searchSpace,1)>size(searchSpace,2))
    searchSpace=searchSpace';
end

end
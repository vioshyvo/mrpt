% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [ index ] = mrpt( data, n0, n_trees )
%The function builds the mrpt index for the given data set.
%   The MRPT index is a vital structure for performing fast nearest
%   neighbor queries in large data sets. The index consists of a collection
%   of random projection trees (rptree.m).
%
%   INPUTS:
%   data - the data set for which the index is built
%   n0 - the maximum number of data objects in a single leaf in any tree
%   n_trees - The number of rp trees used in the index
%
%   OUTPUTS:
%   index - The mrpt index structure, implemented in matlab as a cell array
%   that contains the rptrees that constitute the index.

% If the user does not specify the number of trees and leaf size, use
% default values n0=32, n_trees=32.
if nargin == 1
    n0 = 32;
    n_trees = 32;
end

% Construct the cell array as described.
index = cell(n_trees, 1);
for t = 1:n_trees
    index{t} = rptree(data, n0);
end
end


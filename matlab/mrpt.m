function [ index ] = mrpt( data, n0, n_trees )
%The function builds the mrpt index
%   Detailed explanation goes here
    
    index = {};
    for t = 1:n_trees
       index{t} = rptree(data, n0); 
    end
end


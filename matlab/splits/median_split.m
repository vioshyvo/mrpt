% Splits the ordered projections by the median.
function [splitIndex, splitPoint] = median_split(projectionsOrdered)
    splitIndex=floor(size(projectionsOrdered,2)/2)+1;
    splitPoint = median(projectionsOrdered);
end
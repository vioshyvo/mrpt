function [splitIndex, splitPoint] = dasgupta_random_split(projectionsOrdered)
N = size(projectionsOrdered,2);
splitPoint = min(projectionsOrdered)+((1+2*rand)/4)*(max(projectionsOrdered)-min(projectionsOrdered));
splitIndex=2;
while (splitIndex<N && projectionsOrdered(splitIndex)<splitPoint)
    splitIndex = splitIndex+1;
end
end
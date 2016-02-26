function [splitIndex, splitPoint] = mean_split(orderedProjections)

N = size(orderedProjections,2);
splitPoint = mean(orderedProjections);
splitIndex=2;
while (orderedProjections(splitIndex)<splitPoint && splitIndex<N)
   splitIndex = splitIndex+1; 
end

end
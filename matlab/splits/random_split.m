% DOES NOT BEHAVE WELL IF THE DATA HAS A LOT OF DUPLICATES!!!

function [splitIndex, splitPoint] = random_split(projectionsOrdered)

% Number of projections.
N = size(projectionsOrdered,2);

splitPoint = projectionsOrdered(1)+(projectionsOrdered(N)-projectionsOrdered(1))*rand;

splitIndex=2;

while (splitIndex<N && projectionsOrdered(splitIndex)<splitPoint)
    splitIndex = splitIndex+1;
end

end
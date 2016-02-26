
function [splitIndex, splitPoint] = longest_interval_split(projectionsOrdered)


longest=0;
splitIndex=1;

for ii = ceil((1/4)*size(projectionsOrdered,2)):floor((3/4)*size(projectionsOrdered,2))
    if (projectionsOrdered(ii+1)-projectionsOrdered(ii)>longest)
        longest = projectionsOrdered(ii+1)-projectionsOrdered(ii);
        splitIndex=ii+1;
    end
end
splitPoint = (projectionsOrdered(splitIndex-1)+projectionsOrdered(splitIndex))/2;
end
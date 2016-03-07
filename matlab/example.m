% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

% The script contains a simple test scenario of the MRPT algorithm.

k=10;
n_queries = 100;
n0 = 32;
n_trees = 32;

[trainimgs, trainlbls] = read_mnist('train');
[testimgs, testlbls] = read_mnist('test');
query_indexes = randperm(size(testimgs, 1));
queries = testimgs(query_indexes(1:n_queries),:);

fprintf('Looking for %d nearest neighbors. Averaging over %d queries.\n',k,n_queries);
fprintf('Building index with T=%d, n0=%d.\n',n_trees, n0);

tic;
index = mrpt(trainimgs, n0, n_trees);
fprintf('Index build time %f\n', toc);

tic;
aneighbors = zeros(n_queries, k);
for ii=1:n_queries
    aneighbors(ii, :) = ann(trainimgs, index, queries(ii,:), k);
end
fprintf('Avg approximate query time %1.10f\n', toc/n_queries);

tic;
[~,ordering] = sort(pdist2(queries, trainimgs),2);
tneighbors = ordering(:,1:k);
fprintf('Avg linear query time %1.10f\n', toc/n_queries);

n_correct = zeros(n_queries, 1);
for ii = 1:n_queries
   n_correct(ii) = size(intersect(aneighbors(ii,:),tneighbors(ii,:)),2); 
end
fprintf('Accuracy: %f\n', sum(n_correct)/n_queries);

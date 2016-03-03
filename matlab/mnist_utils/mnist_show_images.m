% A function written to visualize MNIST handwritten digits. The first
% parameter is the image data as a matrix, one image per row. The other
% parameter is the indices of the digits to be visualized.

function [] = mnist_show_images(data, indices)

s = ceil(sqrt(size(indices,2)));

figure()
for ii = 1:size(indices,2)
   subplot(s,s,ii)
   image(reshape(100*data(indices(ii),:),28,28)')
   axis off
end
colormap gray
end
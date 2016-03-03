% Author Teemu Henrikki Pitk?nen <teemu.pitkanen@helsinki.fi>
% University of Helsinki
% Helsinki Institute for Information Technology 2016

function [imgs, lbls] = read_mnist(train_or_test, src_dir)
%Function to read the MNIST dataset from file
% INPUT:
% train_or_test - String, either 'train' or 'test'
% src_dir - The location of the MNIST files
% OUTPUT:
% imgs - the mnist images as 768-dimensional row vectors
% labels - the corresponding labels

if nargin ~=2
    src_dir = '../datasets/mnist/';
    if nargin ~=1
        train_or_test = 'train';
    end
end

if strcmp(train_or_test, 'train')
    img_src = [src_dir 'train-images.idx3-ubyte'];
    lbl_src = [src_dir 'train-labels.idx1-ubyte'];
else
    img_src = [src_dir 't10k-images.idx3-ubyte'];
    lbl_src = [src_dir 't10k-labels.idx1-ubyte'];
end

fid = fopen(img_src, 'r', 'b');
fread(fid, 1, 'int32'); % Magic number
n_imgs = fread(fid, 1, 'int32');
rows = fread(fid, 1, 'int32');
cols = fread(fid, 1, 'int32');
imgs = zeros(n_imgs, rows*cols);
for ii = 1:n_imgs
    imgs(ii,:) = fread(fid,rows*cols, 'uint8');
end
fclose(fid);

fid = fopen(lbl_src, 'r', 'b');
fread(fid, 1, 'int32'); % Magic number
n_lbls = fread(fid, 1, 'int32');
lbls = fread(fid, n_lbls, 'uint8');
fclose(fid);

end

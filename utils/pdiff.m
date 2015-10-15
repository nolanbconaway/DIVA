function B = pdiff(A,DIM)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Compute all pairwise differences between elements in an matrix, 
% operating along dimension DIM. This function accepts matrices of any
% size. 
% 
% Example Usage:
%   A = [ 1 2 3 4
%         5 6 5 7
%         3 4 2 1 ];
% 
%   B = pdiff(A,2)
% 
%   B =
%        1     2     3     1     2     1
%        1     0     2    -1     1     2
%        1    -1    -2    -2    -3    -1
% 
% B now contains all differences between elements within each row. 
% The differences are stored sequentially:
% 
%   B = [ A(:,2) - A(:,1), A(:,3) - A(:,1) , A(:,4) - A(:,1) ,...
%         A(:,3) - A(:,2), A(:,4) - A(:,2), A(:,4) - A(:,3)];
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

%// New permuting dimensions
new_permute = [DIM setdiff(1:ndims(A),DIM)];

% Permuted A and its 2D reshaped version
A_perm = permute(A,new_permute);
A_perm_2d = reshape(A_perm,size(A,DIM),[]);

% Get pairwise index for each dimension
N = size(A,DIM);
[Y,X] = find(bsxfun(@gt,[1:N]',1:N)); % fliplr(nchoosek(1:size(A,dim),2))

% Get size of new permuted array that would have the length of 
% first dimension equal to number of such pairwise combinations 
sz_A_perm = size(A_perm);
sz_A_perm(1) = numel(Y);

% Get the pairwise differences; reshape to a multidimensional array of same
% number of dimensions as the input array
diff_mat = reshape(A_perm_2d(Y,:) - A_perm_2d(X,:),sz_A_perm);

% Permute back to original dimension sequence as the final output
[~,return_permute] = sort(new_permute);
B = permute(diff_mat,return_permute);

return
end

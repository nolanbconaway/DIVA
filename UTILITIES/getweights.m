function [inweights,outweights] = getweights(...
	numfeatures, numhiddens, numcategories, weightrange, weightcenter)


bias=1;

% GENERATE WEIGHTS BETWEEN INPUT AND HIDDEN LAYER
% -----------------------------------------------
inweights = 2 * (rand(numfeatures + bias, numhiddens) - 0.5);
inweights = weightcenter + (weightrange * inweights); 


% GENERATE WEIGHTS BETWEEN HIDDEN AND OUTPUT LAYER
% ------------------------------------------------
outweights = 2 * (rand(numhiddens + bias, numfeatures, numcategories) - 0.5); 
outweights = weightcenter + (weightrange * outweights); 

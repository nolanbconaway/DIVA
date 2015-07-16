function [ps,ssqerror,fweights] = responserule(...
	outputactivations,...	% generated outputs from forward pass
	targetactivations,...	% target activations
	betavalue)				% focusing parameter

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% USAGE
% [ps,ssqerror,fweights] = responserule(outputactivations, targets, betavalue) 	
% 
% DESCRIPTION
% 	This script implements DIVA's response rule for converting output
% 	activations into classification probabilities. It should be used after
% 	calling FORWARDPASS.m to generate output activations.
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% OUTPUT ARGUMENTS
% 	p				probability of classification for target category
% 	ssqerror		error used in the Luce Choice formula
% 	fweights		focusing weights used to weigh ssqerror
% 
% INPUT ARGUMENTS
% 	outputactivations		generated outputs from forward pass
% 	targetactivations		target activations
% 	betavalue				focusing parameter 
% 
%-------------------------------------------------------------------------
numcategories = size(outputactivations, 3);
numfeatures = size(targetactivations, 2);
numstimuli = size(targetactivations, 1);

%-------------------------------------------
% get error on each output channel
ssqerror=outputactivations-repmat(targetactivations,[1,1,numcategories]);
ssqerror=ssqerror.^2;
ssqerror(ssqerror<1e-7) = 1e-7;

%-------------------------------------------
% generate focus weights
diversities = exp(betavalue.*mean(abs(diff(outputactivations,[],3)),3));
diversities(diversities>1e+7) = 1e+7; % prevent Infs
fweights = diversities ./ repmat(sum(diversities,2),[1,numfeatures]);

%  apply focus weights, then get the sum for each category
ssqerror=sum(ssqerror.*repmat(fweights,[1,1,numcategories]),2);
ssqerror=reshape(ssqerror,[numstimuli,numcategories]);

%-------------------------------------------
% get class probability
ssqerror = 1 ./ ssqerror;
ps = ssqerror./repmat(sum(ssqerror,2),[1,numcategories]);

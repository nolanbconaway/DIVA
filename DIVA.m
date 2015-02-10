function result= DIVA(model)
% ----------------------------------------------------------------------------
% DESCRIPTION
%	this script does most of the work for training DIVA. 
%	it creates a result struct containing accuracy over trials/blocks, as
%	well as a log of the the weights learned by diva.
% 
% INPUT ARGUMENTS:
% 	model is a struct that is assumed to contain:
% 		model.numblocks	  		= 160; % number of weight updates
% 		model.numinitials		= 50; % number of randomized divas
% 		model.weightrange		= .5; % range of initial weight values
% 		model.numhiddenunits 	= 2; % # hidden units
% 		model.learningrate   	= .15; % learning rate for gradient descent
% 		model.betavalue	  		= 2.5; % beta parameter for focusing
%	   	model.outputrule = activation rule for output units. {"linear", " sigmoid"}.
% 
%		model.input is an [eg,dimension] matrix of training exemplars
%		model.labels is an integer vector of class labels for each input
% ----------------------------------------------------------------------------
v2struct(model) %unpack input params
rng('shuffle') %get new random seed

%   these are optional editables, currently set at default values
	weightcenter = 0; % mean value of weights
	
	% convert all targets to [0 1] for consistency with sigmoid
	targets = globalscale(inputs,[0 1]);
% ----------------------------------------------------------------------------

result=struct; %initialize the results structure

% initializing some useful variables
numfeatures = size(inputs,2);
numstimuli = size(inputs,1); 
numcategories = length(unique(labels));
numupdates = numblocks*numstimuli;

% set up storage for model performance
training=zeros(numupdates,numinitials);

%   Initializing diva and running the simulation
%   ------------------------------------------------------ % 
for modelnumber = 1:numinitials
	
	%  generating initial weights
	[inweights,outweights] = getweights(numfeatures, numhiddenunits, ...
		numcategories, weightrange, weightcenter);

	%  generating full presentation order
	presentationorder = getpresentationorder(numstimuli,numblocks);
	
	%   iterate over each trial in the presentation order
	%   ------------------------------------------------------ % 
	for trialnumber = 1:numupdates
		currentinput  =  inputs(presentationorder(trialnumber),:);
		currenttarget =  targets(presentationorder(trialnumber),:);
		currentclass  =  labels(presentationorder(trialnumber),:);
		
		%  ------------------- complete forward pass
		[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			FORWARDPASS(inweights,outweights,currentinput,outputrule);
		
		% --- compute classification probability
		p = responserule(outputactivations, currenttarget, betavalue);
		
		% Store classification accuracy
		training(trialnumber,modelnumber)=p(currentclass);
		
		%  ------------------- back propagate error to adjust weights
		classweights = outweights(:,:,currentclass);
		classactivation = outputactivations(:,:,currentclass);
		[outweights(:,:,currentclass), inweights] = BACKPROP(...
			classweights,inweights,classactivation,currenttarget,...
			hiddenactivation,hiddenactivation_raw,inputswithbias,learningrate);
	end
	
% %	 ----TEST PHASES CAN BE ADDED HERE. THIS IS A SAMPLE:
%	 TEST_OUTS = FORWARDPASS(inweights,outweights,TEST_INPUTS,outputrule);
% 	 TEST_CLASSIFY = responserule(TEST_OUTS, TEST_TARGETS, betavalue);
% %	 ----------- END TEST PHASES
end

% store performance in the result struct
result.training=blockrows(mean(training,2),numstimuli)'; 
end

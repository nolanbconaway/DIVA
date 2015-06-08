function...
	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,... % weight matrices
			inputs,... % activations to be passed through the model
			outputrule) % option for activation rule	
				   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% USAGE
% 	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
%		forwardpass(inweights,outweights,inputs,outputrule)
% 
% DESCRIPTION
% 	This completes a forward pass, and returns p(cat),as well as any info 
% 	needed for backprop for each activation. This script can be used for 
% 	trial by trial data, or for a vector of inputs.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% OUTPUT ARGUMENTS
% 	outputactivations			output layer activations
% 	hiddenactivation			hidden layer activations,including bias
% 	hiddenactivation_raw		dot product of inputs and in-hid weights
% 	inputswithbias				input activations, with bias
% 
% INPUT ARGUMENTS
% 	inweights, outweights		weight matrices
% 	inputs (M x N matrix)		activations to be passed through the model
%   outputrule (string)			option for activation rule
% 
%-------------------------------------------------------------------------

numstimuli=size(inputs,1);
numfeatures=size(inputs,2);
numcategories=size(outweights,3);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% input and hidden unit propgation
inputswithbias = [ones(numstimuli,1),inputs]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
hiddenactivation=sigmoid(hiddenactivation_raw);

% adding a value of 1 to represent the bias unit 
hiddenactivation=[ones(numstimuli,1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activaton
outputactivations=zeros(numstimuli,numfeatures,numcategories);
for o = 1:numcategories
	outputactivations(:,:,o)=(hiddenactivation*outweights(:,:,o));
end

% applying output activation rule
if strcmp(outputrule,'sigmoid') 
	outputactivations = sigmoid(outputactivations);
end


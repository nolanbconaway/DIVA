function...
	[p,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,... % weight matrices
			inputs,... % activations to be passed through the model
			targets,... % target output activation values for each input
			outactrule,... % option for activation rule
			betavalue,... % focusing paramater
			currentcategory) % category label that p(a) is evaluated by	
				   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% USAGE
% 	[pCat,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
% 	forwardpass(inweights,outweights,inputpatterns,hiddenactrule,
% 			outactrule,beta,humbleClassify,valueRange,currentcategory)
% 
% DESCRIPTION
% 	This completes a forward pass, and returns p(cat),as well as any info 
% 	needed for backprop for each activation. This script can be used for 
% 	trial by trial data, or for a vector of inputs.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
% OUTPUT ARGUMENTS
% 	p: probabilty of classification as member of category C
% 	outputactivations: output layer activations
% 	hiddenactivation: hidden layer activations,including bias
% 	hiddenactivation_raw: dot product of inputs and in-hid weights
% 	inputswithbias: input activations, with bias
% 
% INPUT ARGUMENTS
% 	inweights,outweights: weight matrices
% 	inputpatterns (M x N matrix): activations to be passed through the model
%   outactrule (string): option for activation rule
% 	beta (0<=x<inf): focusing paramater
% 	currentcategory(integer): category label that pCat is evaluated by	
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
hiddenactivation=logsig(hiddenactivation_raw);

% adding a value of 1 to represent the bias unit 
hiddenactivation=[ones(numstimuli,1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activaton
outputactivations=zeros(numstimuli,numfeatures,numcategories);
for o = 1:numcategories
	outputactivations(:,:,o)=(hiddenactivation*outweights(:,:,o));
end

% applying output activation rule
if strcmp(outactrule,'sigmoid') 
	outputactivations = logsig(outputactivations);
elseif strcmp(outactrule,'clipped') 
	outputactivations = clipvalues(outputactivations,[0 1]);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% caluclate error, focus weights, and classification probabilities

% get error on each output channel
ssqerror=outputactivations-repmat(targets,[1,1,numcategories]);
ssqerror=ssqerror.^2;
ssqerror(ssqerror<1e-7) = 1e-7;

% generate focus weights
diversities = exp(betavalue.*mean(abs(diff(outputactivations,[],3)),3));
fweights = diversities ./ repmat(sum(diversities,2),[1,numfeatures]);

%  apply focus weights, then get the sum for each category
ssqerror=sum(ssqerror.*repmat(fweights,[1,1,numcategories]),2);
ssqerror=reshape(ssqerror,[numstimuli,numcategories]);

% get class probability
ssqerror = 1 ./ ssqerror;
p = ssqerror(:,currentcategory)./sum(ssqerror,2);




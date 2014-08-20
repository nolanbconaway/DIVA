function...
	[p,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,...%weight matrices
			inputpatterns,...%activations to be passed through the model
			hiddenactrule,outactrule,...%option for activation rule
			betavalue,...%focusing paramater
			humbleclassify,valuerange,...option to clip activations
			currentcategory) %category label that p(a) is evaluated by    
                   
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
% 	hiddenactrule,outactrule (string): option for activation rule
% 	beta (0<=x<inf): focusing paramater
% 	humbleclassify (bool),valuerange (vector): option to clip activations
% 	currentcategory(integer): category label that pCat is evaluated by    
% 
%-------------------------------------------------------------------------

numstimuli=size(inputpatterns,1);
numfeatures=size(inputpatterns,2);
numcategories=size(outweights,3);

outputactivations=zeros(numstimuli,numfeatures,numcategories);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% input and hidden unit propgation
inputswithbias = [ones(numstimuli,1),inputpatterns]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
if strcmp(hiddenactrule,'sigmoid') % applying sigmoid;
    hiddenactivation=sigmoid(hiddenactivation_raw);
elseif strcmp(hiddenactrule,'tanh')  %applying tanh
    hiddenactivation=tanh(hiddenactivation_raw);
else hiddenactivation=hiddenactivation_raw;
end

% adding a value of 1 to represent the bias unit 
hiddenactivation=[ones(numstimuli,1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activaton
for o = 1:numcategories
    outputactivations(:,:,o)=(hiddenactivation*outweights(:,:,o));
end

if strcmp(outactrule,'sigmoid') % applying sigmoid
	outputactivations=sigmoid(outputactivations);
elseif strcmp(outactrule,'tanh') %applying tanh
	outputactivations=tanh(outputactivations);
end

% apply clipping at output layer
if humbleclassify
	outputs_for_response= clipvalues(outputactivations,valuerange);
else outputs_for_response = outputactivations;
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% caluclate error, focus weights, and classification probabilities

% get error on each output channel
ssqerror=outputs_for_response-repmat(inputpatterns,[1,1,numcategories]);

% square error and make sure there are no zeros
ssqerror=ssqerror.^2;
ssqerror(ssqerror<1e-7) = 1e-7;

% generate focus weights
if numcategories==2		
	diversities = exp(betavalue.*abs(outputs_for_response(:,:,1)-outputs_for_response(:,:,2)));
	fweights = diversities ./ repmat(sum(diversities,2),[1,numfeatures]);
else fweights=ones(size(inputpatterns));
end

%  apply focus weights, then get the sum for each category
ssqerror=sum(ssqerror.*repmat(fweights,[1,1,numcategories]),2);

% reshape and inverse the error
ssqerror=1./reshape(ssqerror,[numstimuli,numcategories]);

% get probability
p=ssqerror(:,currentcategory)./sum(ssqerror,2);


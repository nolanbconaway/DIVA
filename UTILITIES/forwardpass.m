function...
	[pCat,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		forwardpass(inweights,outweights,...%weight matrices
			inputpatterns,...%activations to be passed through the model
			hiddenactrule,outactrule,...%option for activation rule
			beta,...%focusing paramater
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
% 	pCat: probabilty of classification as member of category C
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

numpatterns=size(inputpatterns,1);
numcategories=size(outweights,3);

outputactivations=zeros(size(inputpatterns,1),...
	size(inputpatterns,2),size(outweights,3));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% input and hidden unit propgation
inputswithbias = [ones(size(inputpatterns,1),1),inputpatterns]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
if strcmp(hiddenactrule,'sigmoid') % applying sigmoid;
    hiddenactivation=sigmoid(hiddenactivation_raw);
elseif strcmp(hiddenactrule,'tanh')  %applying tanh
    hiddenactivation=tanh(hiddenactivation_raw);
else hiddenactivation=hiddenactivation_raw;
end

% adding a value of 1 to represent the bias unit 
hiddenactivation=[ones(size(hiddenactivation,1),1),hiddenactivation];

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% get output activaton
for o=1:numcategories
    outputactivations(:,:,o)=(hiddenactivation*outweights(:,:,o));
end
if strcmp(outactrule,'sigmoid') % applying sigmoid
	outputactivations=sigmoid(outputactivations);
elseif strcmp(outactrule,'tanh') %applying tanh
	outputactivations=tanh(outputactivations);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% caluclate error, focus weights, and classification probabilities
if numcategories==2;%focus weights    
    fweights=exp(beta*(abs(...
		outputactivations(:,:,1)-outputactivations(:,:,2))-range(valuerange)));
else fweights=ones(size(inputpatterns));
end

% get error on each output channel
if humbleclassify
	ssqerror=humbleTeach(outputactivations,valuerange)-repmat(inputpatterns,[1,1,numcategories]);
else ssqerror=outputactivations-repmat(inputpatterns,[1,1,numcategories]);
end
% square error and apply focus weights, then get the sum
ssqerror=sum((ssqerror.^2).*repmat(fweights,[1,1,numcategories]),2);

% cap error at realmax to prevent NaN and shape result in 2D
ssqerror(ssqerror>realmax)=realmax;
ssqerror=reshape(1./ssqerror,[numpatterns,numcategories]);

% get probability
pCat=ssqerror(:,currentcategory)./sum(ssqerror,2);

clear in_act hidz hida fweights ssqerror


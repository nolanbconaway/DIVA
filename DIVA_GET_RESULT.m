function result= DIVA_GET_RESULT(diva,inputs,labels)

% ----------------------------------------------------------------------------
% DESCRIPTION
%	this script does most of the work for training DIVA. 
%	it creates a result struct containing accuracy over trials/blocks, as
%	well as a log of the the weights learned by diva.

% INPUT ARGUMENTS:
% 	diva is a struct that is assumed to contain:
% 		diva.numUpdates = 160; % number of weight updates
% 		diva.numInitials = 50; % number of randomized divas
% 		diva.weightRange = .5; % range of inital weight values
% 		diva.numHiddenUnits = 2; % # hidden units
% 		diva.learningRate = .15; % learning rate for gradient descent
% 		diva.betaValue = 2.5; % beta parameter for focusing
% 		diva.clipValues=[true, true]; %clip values for [classify,backprop]
% 
%	input is an [eg,dimension] matrix containing a block of unique examples
%	labels is an integer vector containing category labels for each row in input
% ----------------------------------------------------------------------------

%   these are optional editables
	hiddenactrule = 'sigmoid'; % which activation rule?
	outputactrule = 'linear'; % options: 'linear', 'sigmoid', 'tanh'
	valueRange=[floor(min(min(inputs))),ceil(max(max(inputs)))];
	weightCenter=0; % mean value of weights
    humbleClassify = true; % clip activations at a 
    humbleLearn =  true;   % min and maximum value?
	
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

result=struct; %initialize the results structure
v2struct(diva) %unpack input params

% initializing some useful variables
numDims = size(inputs,2);
numStim = size(inputs,1); 
numCats = length(unique(labels));

train_accuracy=zeros(numUpdates,numInitials);

% Initializing diva and running the simulation
for modelnumber = 1:numInitials
    
%     generating initial weights
    inWeights = getWeights(numDims, numHiddenUnits, weightRange, weightCenter);
    outWeights=zeros(numHiddenUnits+1,numDims,numCats);
    for o=1:numCats;
        outWeights(:,:,o)= getWeights(numHiddenUnits, numDims, weightRange, weightCenter);
    end
    
    %     generating full example set
    [networkinput, networklabels] = orderexampleset(numUpdates,inputs,labels);
    
    
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%   this loop iterates over the full example set, 
%   updating the weights after every example
    for trialnumber = 1:numUpdates
% 		define current item being pass through the model
		currentinput=networkinput(trialnumber,:);
		currentcategory=networklabels(trialnumber);
		
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % Forward propagation of input activations
        [pCat,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			forwardpass(inWeights,outWeights,currentinput,hiddenactrule,outputactrule,...
                betaValue,humbleClassify,valueRange,currentcategory);
        train_accuracy(trialnumber,modelnumber)=pCat;
		

	% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % Back-propagating the activations
		
%       	obtain error on the output units
		if humbleLearn
			outputderivative=2*(humbleTeach(...
				outputactivations(:,:,currentcategory),valueRange) - currentinput);
		else outputderivative=2*(outputactivations(:,:,currentcategory) - currentinput);
		end

%			obtain error on the hidden units
		hiddenderivative=outputderivative*outWeights(:,:,currentcategory)';
		if strcmp(hiddenactrule,'sigmoid') % applying sigmoid;
			hiddenderivative=hiddenderivative(:,2:end).*sigmoidGradient(hiddenactivation_raw);
		elseif strcmp(hiddenactrule,'tanh') %applying tanh
			hiddenderivative=hiddenderivative(:,2:end).*tanhGradient(hiddenactivation_raw);
		else hiddenderivative=hiddenderivative(:,2:end).*...
				(hiddenactivation_raw.*(1-hiddenactivation_raw));
		end 

%       	gradient descent
		outWeights(:,:,currentcategory) = gradientDescent(learningRate,...
			hiddenactivation,outputderivative,outWeights(:,:,currentcategory));
		inWeights = gradientDescent(learningRate,...
			inputswithbias,hiddenderivative,inWeights);    
		     
%     Clearing out some variables for the next iteration
        clear pCat outputactivations hiddenactivation hiddenactivation_raw ...
			inputswithbias outputderivative hiddenderivative ...
			currentinput currentcategory
	end
% 	[pCat,outputactivations,hiddenactivation] = forwardpass(inWeights,outWeights,inputs,hiddenactrule,outputactrule,...
%                 betaValue,humbleClassify,valueRange,1);
%       [inputs,hiddenactivation]
%     Clearing out some variables for the next initialization
    clear network_input network_labels Input_Hidden_wts outWeights
end

% store perfomance in the result struct
result.trialByTrialAccuracy=mean(train_accuracy,2);

if numStim<=numUpdates% get mean accuracy by block
    result.blockByBlockAccuracy=returnblocks(mean(train_accuracy,2),numStim); 
end    
end

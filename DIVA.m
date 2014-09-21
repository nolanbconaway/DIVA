function result= DIVA(diva,inputs,labels)

% ----------------------------------------------------------------------------
% DESCRIPTION
%	this script does most of the work for training DIVA. 
%	it creates a result struct containing accuracy over trials/blocks, as
%	well as a log of the the weights learned by diva.

% INPUT ARGUMENTS:
% 	diva is a struct that is assumed to contain:
% 		diva.numblocks = 160; % number of weight updates
% 		diva.numinitials = 50; % number of randomized divas
% 		diva.weightrange = .5; % range of inital weight values
% 		diva.numhiddenunits = 2; % # hidden units
% 		diva.learningrate = .15; % learning rate for gradient descent
% 		diva.betavalue = 2.5; % beta parameter for focusing
% 
%	input is an [eg,dimension] matrix containing a block of unique examples
%	labels is an integer vector containing category labels for each row in input
% ----------------------------------------------------------------------------

%   these are optional editables, currently set at default values
	hiddenactrule = 'sigmoid'; % which activation rule?
	outputactrule = 'linear'; % options: 'linear', 'sigmoid', 'tanh'
	valuerange=[floor(min(min(inputs))),ceil(max(max(inputs)))];
	weightcenter=0; % mean value of weights
    humbleclassify = true; % clip activations at a 
    humblelearn =  true;   % min and maximum value?	
% ----------------------------------------------------------------------------

result=struct; %initialize the results structure
v2struct(diva) %unpack input params

% initializing some useful variables
numfeatures = size(inputs,2);
numstimuli = size(inputs,1); 
numcategories = length(unique(labels));
numupdates = numblocks*numstimuli;


training=zeros(numupdates,numinitials);

%   Initializing diva and running the simulation
%   ------------------------------------------------------ % 
for modelnumber = 1:numinitials
    
    %  generating initial weights
    [inweights,outweights] = getweights(numfeatures, numhiddenunits, ...
		numcategories, weightrange, weightcenter);

    %  generating full presentation order
    presentationorder = getpresentationorder(numstimuli,numblocks);
    networkinput=inputs(presentationorder,:);
    networklabels=labels(presentationorder,:);
    
    %   iterate over each trial in the presentation order
    %   ------------------------------------------------------ % 
	for trialnumber = 1:numupdates
        
		currentinput=networkinput(trialnumber,:);
		currentcategory=networklabels(trialnumber);
		
        [pCat,outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			FORWARDPASS(inweights,outweights,currentinput,hiddenactrule,outputactrule,...
                betavalue,humbleclassify,valuerange,currentcategory);
        
        training(trialnumber,modelnumber)=pCat;
		
        
        %   Back-propagating the activations
        %   ------------------------------------------------------ % 
        
        %  obtain error on the output units
        if humblelearn
			outputderivative=2*(clipvalues(...
				outputactivations(:,:,currentcategory),valuerange) - currentinput);
        else outputderivative = 2*(outputactivations(:,:,currentcategory) - currentinput);
        end

        %  obtain error on the hidden units
		hiddenderivative=outputderivative*outweights(:,:,currentcategory)';
        if strcmp(hiddenactrule,'sigmoid') % applying sigmoid;
			hiddenderivative=hiddenderivative(:,2:end).*sigmoidGradient(hiddenactivation_raw);
        elseif strcmp(hiddenactrule,'tanh') %applying tanh
			hiddenderivative=hiddenderivative(:,2:end).*tanhGradient(hiddenactivation_raw);
        else hiddenderivative=hiddenderivative(:,2:end).*...
				(hiddenactivation_raw.*(1-hiddenactivation_raw));
        end 

        %  gradient descent
		outweights(:,:,currentcategory) = gradientDescent(learningrate,...
			hiddenactivation,outputderivative,outweights(:,:,currentcategory));
		inweights = gradientDescent(learningrate,inputswithbias,...
			hiddenderivative,inweights);    
	end
end

% store perfomance in the result struct
result.training=blockrows(mean(training,2),numstimuli)'; 
end

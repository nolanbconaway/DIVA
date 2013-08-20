function [examples, labs]= orderexampleset(numUpdates,inputs,labels)
                            
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% this script takes in the number of updates and number of stimuli and 
% generates a randomly ordered, permuted exampleset for DIVA. if the number
% of stimuli does not evenly divide into the number of updates (e.g., when
% numStim=8 and numUpdates/=8,16,24...), then the remaining exampleset is 
% created with a random sample of the possible egs.
% 
% numUpdates = number of weight updates
% inputs = input activations
% labels = category labels for the inputs
numStim = size(inputs,1);
blocks=floor(numUpdates/numStim);

if blocks>=1
	egs=[];
% 	generate permuted blocks
	for block=1:blocks
		egs=cat(1,egs,randperm(numStim)');
	end
	
% 	add remaining egs
	if mod(numUpdates,numStim)~=0
		remain=numUpdates-(numStim * blocks);
		partialblock=randperm(numStim)';
		egs=cat(1,egs,partialblock(1:remain));
	end
	
else
	egs=randperm(numStim);
	egs=egs(1:numUpdates);
end

examples = inputs(egs,:);
labs = labels(egs,:);
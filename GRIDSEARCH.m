
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %		_	 _	 	 _	   _	  _	    _	   _	 _		 	  % %
% %	   (c).-.(c)	(c).-.(c)	 (c).-.(c)	  (c).-.(c)			  % %
% %		/ ._. \	 	 / ._. \	  / ._. \	   / ._. \		 	  % %
% %	  __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__	   	  % %
% %	 (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)	  	  % %
% %		|| D ||	  	 || I ||	  || V ||	   || A ||		 	  % %
% %	  _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._	   	  % %
% %	 (.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)	  	  % %
% %	  `-'	 `-'  `-'	   `-'  `-'	 	`-'  `-'	 `-'	   	  % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Use this script to store performance for a range of parameterizations
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 

% set search range
numhiddenunits = [2,3,4];	  % # hidden units
learningrate = [0.1:0.1:0.5]; % learning rate for gradient descent
weightrange = [1, 2, 3];	  % range of initial weight values
betavalue = [0,5,10,50];	  % focusing multiplier
parameterlist = allcomb(numhiddenunits,learningrate,weightrange,betavalue);
numparamconfigs = size(parameterlist,1); 

% configuration for console updates
updatefrequency = ceil(numparamconfigs*0.05);

% initialize network design and set parameters
model =  struct;
	model.numblocks = 32;   % number of runs through the training set
	model.numinitials = 10; % number of randomized divas to be averaged across
	model.outputrule = 'sigmoid'; % {'clipped', 'sigmoid' }
	
% iterate across param values
training = zeros(model.numblocks,6,numparamconfigs);
for paramnum = 1:numparamconfigs
	
% 	set current params
	currentparams = parameterlist(paramnum,:);
		model.numhiddenunits = currentparams(1); 
		model.learningrate = currentparams(2); 
		model.weightrange = currentparams(3);
		model.betavalue = currentparams(4);
		
% 	iterate across shj types
	for shj = 1:6
		[model.inputs, model.labels] = SHJINPUTS(shj);
		result = DIVA(model);
		training(:,shj,paramnum) = result.training;
	end
	
% 	print current progress to console
	if mod(paramnum,updatefrequency)==1
		percentcomplete = 100*(paramnum/numparamconfigs);
		disp([num2str(percentcomplete,6) ' percent complete.'])
	end			
end

disp('---------COMPLETE---------')
save('diva.mat','training','parameterlist')




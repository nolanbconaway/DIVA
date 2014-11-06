
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
% Use this script to initialize the DIVA model and begin a simulation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 16;	  % number of runs through the training set
	model.numinitials = 5;	 % number of initializations to average 
	model.weightrange = 0.5;   % range of initial weight values
	model.numhiddenunits = 2;  % # hidden units
	model.learningrate = 0.25; % learning rate for gradient descent
	model.betavalue = 10;	  % beta parameter for focusing
	model.outputactrule = 'sigmoid'; % {'clipped', 'sigmoid' }

% iterate across shj types
training = zeros(model.numblocks,6);
for shj = 1:6
	
% 	set current category structure
	[model.inputs, model.labels] = SHJINPUTS(shj);
	
% 	run simulation
	result = DIVA(model);
	
% 	add result to training data
	training(:,shj) = result.training;
end
	
disp(training)



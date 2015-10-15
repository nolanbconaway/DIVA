
% initialize the search path
clear;close;clc;
addpath([pwd,'/utils/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 20;	  % number of runs through the training set
	model.numinitials = 5;	 % number of initializations to average 
	model.weightrange = 1;   % range of initial weight values
	model.numhiddenunits = 7;  % # hidden units
	model.learningrate = 0.15; % learning rate for gradient descent
	model.betavalue = 5;	  % beta parameter for focusing
	model.outputrule = 'sigmoid'; % {'linear', 'sigmoid' }

% 	load model inputs
	load shj
	model.inputs = stimuli;
	
	
% iterate across shj types
training = zeros(model.numblocks,6);
for shj = 1:6
	
% 	set current category structure
	model.labels = assignments(:,shj);
	
% 	run simulation
	result = DIVA(model);
	
% 	add result to training data
	training(:,shj) = result.training;
	
end
	
disp(training)

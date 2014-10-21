
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %        _     _      _     _      _     _      _     _         % %
% %       (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)        % %
% %        / ._. \      / ._. \      / ._. \      / ._. \         % %
% %      __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__       % %
% %     (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)      % %
% %        || D ||      || I ||      || V ||      || A ||         % %
% %      _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._       % %
% %     (.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)      % %
% %      `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'       % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Use this script to initalize the DIVA model and begin a simulation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 16; % number of runs through the training set
	model.numinitials = 1; % number of randomized divas to be averaged across
	model.weightrange = 0.5; % range of inital weight values
	model.numhiddenunits = 2; % # hidden units
	model.learningrate = 0.25; % learning rate for gradient descent
	model.betavalue = 10;
	
% iterate across shj types
training = zeros(model.numblocks,6);
for shj = 1:6
	
% 	set current category structre
	[inputs,labels]=SHJINPUTS(shj);
	
% 	run simulation
	result = DIVA(model,inputs,labels);
	
% 	add result to training data
	training(:,shj) = result.training;
end
	
disp(training)



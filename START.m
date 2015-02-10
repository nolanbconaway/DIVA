
% initialize the search path
clear;close;clc;
addpath([pwd,'/utils/']); 

% initialize network design and set parameters
model =  struct;
	model.numblocks = 20;	  % number of runs through the training set
	model.numinitials = 5;	 % number of initializations to average 
	model.weightrange = 0.5;   % range of initial weight values
	model.numhiddenunits = 3;  % # hidden units
	model.learningrate = 0.25; % learning rate for gradient descent
	model.betavalue = 10;	  % beta parameter for focusing
	model.outputrule = 'sigmoid'; % {'linear', 'sigmoid' }

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

% --- PLOTTING RESULTS
figure
for i = 1:6
	plot(training(:,i),'--k')
	text(1:model.numblocks,training(:,i),num2str(i),...
		'horizontalalignment','center','fontsize',15)
	hold on
end
axis([0.5 model.numblocks+0.5 0 1])
axis square
set(gca','ygrid','on')


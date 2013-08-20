close;clear;clc; format shortg
addpath([pwd,'/UTILITIES/']); 

% -------------------------------------------------------------------
% this script runs a grid search over divas six paramaters.
% as a demo, it fits to the shj type of your choice, but is problem-general
% -------------------------------------------------------------------

% pick your shj type
shjtype=2;

% determine data that needs to be fit
behavioraldata=nosofsky_shjdata(shjtype);
[inputs,labels]=SHJINPUTS(shjtype);

% -------------------------------------------------------------------
%  Define search range
hiddenunits=1:3;
leanringrate=.05:.05:.1;
betavalue=[0 20 50];
weightrange=[.5 1 1.5];
humbleclassify=[true,false];
humblebackprop=[true,false];

% -------------------------------------------------------------------

% create a list of every combination of the above values
parameterlist=allcomb(hiddenunits,leanringrate,betavalue,...
	weightrange,humbleclassify,humblebackprop);
numParamConfigurations=size(parameterlist,1);

% -------------------------------------------------------------------
% initialize diva's design
diva=struct;
	diva.numUpdates = size(inputs,1)*length(behavioraldata); % number of weight updates
	diva.numInitials = 10; % number of randomized divas to be averaged across
% ------------------------------------------------------------------- 

% -------------------------------------------------------------------
% iterate over parameters
fits=zeros(numParamConfigurations,1);
for paramnum=1:numParamConfigurations
	% 	get current parameter configuration
	params=parameterlist(paramnum,:);
		diva.numHiddenUnits = params(1); % # hidden units
		diva.learningRate = params(2); % learning rate for gradient 
		diva.betaValue = params(3); % beta parameter for focusing
		diva.weightRange = params(4); % range of inital weight values
		diva.clipValues=[params(5), params(6)]; %classify,backprop
		
%   run simulation
	result = DIVA_GET_RESULT(diva,inputs,labels);
  
%   calculate and store fit (in SSD)
	fits(paramnum)=sum((result.blockByBlockAccuracy-behavioraldata).^2);
	
% 	update the console and save current fits
	if mod(paramnum,10)==1 % every 10 fits
		disp([fits(paramnum),parameterlist(paramnum,:)]);
		save('fits.mat','fits')
	end
	clear result
end

% save fits and paramaters
save('fits.mat','fits','parameterlist')

% diaplay top paramaters
clc;
disp([fits,parameterlist])

clear numParamConfigurations model shjtype behavioraldata

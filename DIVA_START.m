% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % <3 <3 <3 WELCOME TO DIVA <3 <3 <3 % % % % % % % % %
% % % % % % % % % 
% % % % % % % % % these functions assume 
% % % % % % % % %       1) that there is only 1 hidden layer
% % % % % % % % %       2) Weight updates are trial based
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Use this script to initalize the DIVA model and begin the simulation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% initialize the search path
clear;close;clc;
addpath([pwd,'/UTILITIES/']); 
diva=struct;

% % % % % % % % % % % % % % % % % % % % % 
% % NETWORK PARAMETERS & ARCHITECTURE % % 
% % % % % % % % % % % % % % % % % % % % %
shjtype=6;
[inputs,labels]=SHJINPUTS(shjtype);

diva.numUpdates = 8*25; % number of weight updates
diva.numInitials = 1; % number of randomized divas to be averaged across
diva.weightRange = .5; % range of inital weight values
diva.numHiddenUnits = 2; % # hidden units
diva.learningRate = 0.35; % learning rate for gradient descent
diva.betaValue = 0; % beta parameter for focusing


% this passes the parameters to the training scripts.
result = DIVA_GET_RESULT(diva,inputs,labels);
result
% get fit value
simulation=result.blockByBlockAccuracy
behavioral=nosofsky_shjdata(shjtype);
fitvalue=sum((behavioral-simulation).^2)

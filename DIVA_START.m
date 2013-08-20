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

[inputs,labels]=SHJINPUTS(1);

diva.numUpdates = 160; % number of weight updates
diva.numInitials = 10; % number of randomized divas to be averaged across
diva.weightRange = .5; % range of inital weight values
diva.numHiddenUnits = 2; % # hidden units
diva.learningRate = .15; % learning rate for gradient descent
diva.betaValue = 120; % beta parameter for focusing

% clip activations at a min and maximum value?
diva.clipValues=[true, true]; %classify,backprop

% this passes the parameters to the training scripts.
result = DIVA_GET_RESULT(diva,inputs,labels);
result.blockByBlockAccuracy
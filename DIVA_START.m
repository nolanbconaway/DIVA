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
[inputs,labels]=SHJINPUTS(4);

diva.numblocks = 32; % number of runs through the training set
diva.numinitials = 1; % number of randomized divas to be averaged across
diva.weightrange = 0.5; % range of inital weight values
diva.numhiddenunits = 2; % # hidden units
diva.learningrate = 0.25; % learning rate for gradient descent

diva.betavalue = 100; % beta parameter for focusing


% this passes the parameters to the training scripts.

result = DIVA_GET_RESULT(diva,inputs,labels);
disp(result.training)

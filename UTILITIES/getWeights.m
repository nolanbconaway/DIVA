function wts = getWeights(layer1_units, layer2_units, weightRange,center)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% this script generates weights for neural networks; simply provide the
% number of sending units and the number of recieveing units. weights are 
% organized such that each sending unit has its own row, and each 
% receiving unit has its own column. This script assumes full connectivity
% this function also assumes that a bias unit is used. if you don't want
% one, simply change the bias value to 0
% 
% layer1_units = sending units
% layer2_units = reciving units
% weightRange = range of values
% center = 0 means that weights will be centered around 0, 
%           1 = center of .5


bias=1;
wts = weightRange * (2 * ((rand((layer1_units + bias), layer2_units) - .5)));
function wts = getWeights(layer1_units, layer2_units, weightRange, center)

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
% center = center of weight distribution


bias=1;

wts=rand((layer1_units + bias), layer2_units) - 0.5; %interval [-0.5 : 0.5]
wts=2*wts; %interval [-1 : 1]
wts = weightRange * wts; % interval [-weightRange : weightRange]

wts=center+wts;
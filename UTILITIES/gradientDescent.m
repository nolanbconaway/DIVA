function [updated_wts] = gradientDescent(lrnRate,input,delta,wts)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % this script conducts gradient descent on the parameters provided above
% % 
% % lrnrate = obvious
% % input = incoming activations (i.e, from layer n-1, or, x)
% % delta =  error on target values (may be backpropagated or not).
% % wts = obvious

%     compute partial derivative
    bigDelta=input' * delta;

%     compute the value of the weight update
    wtUpdate = lrnRate*bigDelta;

%     execute the update
    updated_wts = wts - (wtUpdate);



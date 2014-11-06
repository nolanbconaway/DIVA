% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %             _     _      _     _      _     _      _     _              % %
% %            (c).-.(c)    (c).-.(c)    (c).-.(c)    (c).-.(c)             % %
% %             / ._. \      / ._. \      / ._. \      / ._. \              % %
% %           __\( Y )/__  __\( Y )/__  __\( Y )/__  __\( Y )/__            % %
% %          (_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)(_.-/'-'\-._)           % %
% %             || D ||      || I ||      || V ||      || A ||              % %
% %           _.' `-' '._  _.' `-' '._  _.' `-' '._  _.' `-' '._            % %
% %          (.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)(.-./`-'\.-.)           % %
% %           `-'     `-'  `-'     `-'  `-'     `-'  `-'     `-'            % %
% %                                                                         % %
% %           Written by Nolan Conaway (nconawa1@binghamton.edu).           % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

This set of scripts runs a minimal version of the DIVA model of category 
learning (Kurtz, 2007). It is written in MATLAB, and is currently set up to 
simulate the six-types problem of Shepard, Hovland, & Jenkins (1961)--though 
it should generalize to any dataset. There are a variety of utility scripts, 
and a few important ones:

(1) START.m can be used to test DIVA using particular parameter sets.
(2) GRIDSEARCH.m can be used to store model performance across a range of 
	parameters
(3) DIVA.m uses a provided architecture to train a network on a set of inputs 
	and category assignments.
(4) FORWARDPASS.m propagates input activations through the network and returns
	measures of network performance

Simulations are run by executing the START pr GRIDSEARCH scripts. All 
simulations begin by passing a model struct to the DIVA script. At a minimum,
'model' needs to include:
	model.numblocks 		number of weight updates
	model.numinitials 		number of randomized divas
	model.weightrange   	range of initial weight values
	model.numhiddenunits 	# hidden units
	model.learningrate   	learning rate for gradient descent
	model.betavalue      	beta parameter for focusing
	model.outputactrule     output rule option:  {"clipped", " sigmoid"}.
    model.input  			an [eg,dimension] matrix of training exemplars
    model.labels 			a column vector of class labels for each input

For almost all situations, inputs should be scaled to [-1 +1]. However, the
target activations should be scaled to [0 1], in order to permit logistic 
output units. By default, the program automatically computes targets as scaled
versions of the inputs. This is done in DIVA.m

By default, DIVA uses clipped linear output units for continuous datasets, and
logistic outputs for binary datasets. This option is set using 
model.outputactrule (set to either 'clipped' or 'sigmoid'). When clipped
linear outputs are used, SSE will be the error measure. Cross-entropy error
is used for logistic outputs. Hidden units are always logistic.

DIVA.m will train the network and return a result struct. As-is, 'result' contains only training accuracy for each initialization at each training block. Additional measures, such as test phase classification, can be added. 
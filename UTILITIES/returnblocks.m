function block_accuracy= returnblocks(trainingdata,numstim)

% takes in a vector of trial accuracy values and averages it across
% blocks, where the block size is defined by numstim

numblocks=floor(length(trainingdata)/numstim);

for b=0:numblocks-1
    firstpos=1+(b*numstim);
    lastpos=(b+1)*numstim;
    block_accuracy(b+1,1)=mean(trainingdata(firstpos:lastpos));
    clear firstpos lastpos
end

if length(trainingdata)>(numblocks*numstim)
    remainder=mean(trainingdata((numblocks*numstim)+1:end));
    block_accuracy=[block_accuracy;remainder];
end
clear numblocks trainingdata numstim remainder
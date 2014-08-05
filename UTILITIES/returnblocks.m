function block_accuracy= returnblocks(trainingdata,numstim)

if any(size(trainingdata)==1)
    block_accuracy = mean(reshape(trainingdata,numstim,[]));
else error('attempting to block aggregate a matrix')
end


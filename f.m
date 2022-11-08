function oobMCR = f(hparams, X, Y)
opts=statset('UseParallel',true);
numTrees=150;
A=TreeBagger(numTrees,X,Y,'method','classification','OOBPrediction','on','Options',opts,...
    'MinLeafSize',hparams.minLS,'NumPredictorstoSample',hparams.numPTS);
oobMCR = oobError(A, 'Mode','ensemble');
end
%code extracted from https://uk.mathworks.com/matlabcentral/answers/347547-using-bayesopt-for-treebagger-classification#answer_273095
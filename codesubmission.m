tic
%Running time: 60 seconds
%I have commented out the hyperparameter tuning
%IMPORTING DATA AND PRE-PROCESSING
rng(765)
train = readtable('train.csv');
test = readtable('test.csv');
%convert columns into categorical
train = convertvars(train,{'job','marital', ...
   'housing','loan','contact','education','day_of_week', ...
   'month','poutcome'},'categorical');
test = convertvars(test,{'job','marital', ...
   'housing','loan','contact','education', 'day_of_week', ...
   'month','poutcome'},'categorical');
%split to X,Y
Xtrain = train(:,1:18);
Ytrain = train(:,19);
Xtest = test(:,1:18);
Ytest = test(:,19);

%Create Dummies
%we use the dummytable.m file [included] [1]
Xtest = dummytable(Xtest);
Xtrain = dummytable(Xtrain);

%normalize train and use values for test
[Xtrain,c,s]= normalize(Xtrain);
Xtest = normalize(Xtest,"center",c,'scale',s);

%convert tables to array – helps with glm functions.
Ytrain = table2array(Ytrain);
Xtrain = table2array(Xtrain);
Ytest = table2array(Ytest);
Xtest = table2array(Xtest);

%FIT BASE LOGISTIC MODEL
model_base = glmfit(Xtrain,Ytrain,'Binomial');
%get predictions
model_base_prob = glmval(model_base,Xtrain,"logit");
model_base_test = glmval(model_base,Xtest,"logit");
[base_a, base_b, base_c, AUC_base] = perfcurve(Ytrain, ...
    model_base_prob,1);
[base_at, base_bt, base_ct, AUC_base_test] = perfcurve(Ytest, ...
    model_base_test,1);
%confusion matrices
%converting prob. to scores
ytrainbase = (model_base_prob>=0.5);
ytestbase = (model_base_test>=0.5);
%confusion matrices
confusionmat(Ytrain, double(ytrainbase))
confusionmat(logical(Ytest),ytestbase)

%LASSO% [code adapted form [2]]
%LASSO LAMBDA TUNING
%intial lamda search
%[B1,FitInfo1] = lassoglm(Xtrain,Ytrain,'binomial',...
    %'NumLambda',25,'CV',10);
%lassoPlot(B1,FitInfo1,'PlotType','CV');
% ylim([-0.5,0.5])
% xlim([0.5,-0.5])
% Lambda = logspace(-200,1000,1000);
%best Lamdba value = 0.0062
%intial lamda search
Lambda =  0.0062;
[B,FitInfo] = lassoglm(Xtrain,Ytrain,'binomial','Lambda',0.0062,'CV',10);
toc
%optimizing lambda (iterative search)

%look up coefficients
idxLambda1SE = FitInfo.Index1SE;
B0 = FitInfo.Intercept(idxLambda1SE);
coef = [B0; B(:,idxLambda1SE)];
fprintf('The coefficients are')
%predict on train
model_lasso_prob = glmval(coef,Xtrain,'logit');
[lass_a, lass_b, lass_c, AUC_lasso] = perfcurve(Ytrain, ...
    model_lasso_prob,1);
AUC_lasso;
%predict on test
model_lasso__test_prob = glmval(coef,Xtest,'logit');
[lass_at, lass_bt, lass_ct, AUC_test_lasso, OPTROClasso] = perfcurve(Ytest, ...
    model_lasso__test_prob,1);
AUC_test_lasso;
%get labels
ytrainlasso = (model_lasso_prob>=0.5);
ytestlasso = (model_lasso__test_prob>=0.5);
%confusion matrices
confusionmat(Ytrain, double(ytrainlasso))
confusionMat_lasso = confusionmat(logical(Ytest),ytestlasso)
confusionchart(logical(Ytest),ytestlasso)

%prediction hisogram
hist(model_lasso__test_prob,30)
xlim([0.2,1])
xlabel('Mean Predicted Probability') 
ylabel('Counts')
title('Logistic Regression Reliability Diagram ')

%set X and Y for convenience
X = Xtrain;
Y = Ytrain;

%HYPERPARAMETER OPTIMIZATION 
%code extracted from [3]
%We use Baysian Optimization to find the optimal hyperparameters of minLS
%and numPTS. Function file f.m [included] is required to use the below snippet.
%We set the number of trees as 150 in f.m
%tic
% minLS = optimizableVariable('minLS',[1,20],'Type','integer');
% numPTS = optimizableVariable('numPTS',[1,58],'Type','integer');
% hyperparametersRF = [minLS;numPTS];
% rng(76);
% fun = @(hyp)f(hyp,X,Y);
% results = bayesopt(fun,hyperparametersRF);
% besthyperparameters = bestPoint(results);
% toc
%Elapsed time is 591.113800 seconds.
% the values are minLs = 20 and numPTS = 10

%FINAL MODEL Random Forest
%predict hyperparameters on final model
Mdl = TreeBagger(150,Xtrain,Ytrain,'Method','classification',...
    'MinLeafSize', 20,...
    'OOBPredictorImportance', 'on',...
    'NumPredictorstoSample',10);
%plot feature importance
figure
bar(Mdl.OOBPermutedPredictorDeltaError)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
%find the most important features(0.85 arbirtraly chosen)
idxvar = find(Mdl.OOBPermutedPredictorDeltaError>0.85);
%get training AUC
%[labels_RDtrain,scores_RDtrain] =  Mdl.predict(Xtrain);
%[ard,brd,~,AUC_RDtrain] = perfcurve((Ytrain), ...
    %scores_RDtrain(:,2),1);
%get test AUC
[labels_RDtest,scores_RDtest] =  Mdl.predict(Xtest);
[atrd,btrd,~,AUC_RDtest,OPTROCRD] = perfcurve((Ytest), ...
    scores_RDtest(:,2),1);

[labels_RDtrain,scores_RDtrain] = oobPredict(Mdl);
[ard,brd,crd,AUC_RDtrain] = perfcurve(Mdl.Y,scores_RDtrain(:,2),1);

%confusion chart
% labels_RDtrain = str2num(cell2mat(labels_RDtrain));
% labels_RDtest = str2num(cell2mat(labels_RDtest));
%train and test
% confusionchart(((Ytrain)), labels_RDtrain)
confusionchart(Ytest,str2double(labels_RDtest))
title('Random Forest Confusion Matrix')
RDconfusion = confusionmat(logical(Ytest),logical(str2double(labels_RDtest)))


%print final AUC of both models
fprintf(['The training AUC of the logistic model is %.2f\n and the test' ...
    ' is %.2f\n'], ...
    AUC_lasso,AUC_test_lasso)
fprintf(['The training AUC of the random forest model is is %.2f\n and the test' ...
    ' is %.2f\n'], ...
    AUC_RDtrain,AUC_RDtest)

%AUC curves
plot(ard,brd)
hold on
plot(atrd,btrd)
plot(lass_a,lass_b)
plot(lass_at,lass_bt)
legend('Random Forest Train','Random Forest Test','Logistic Train','Logistic Test','Location','Best')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve Comparisons')
hold off
toc

%REFERENCES
%[1]James Wiken (2021). Mathematical Modeling with MATLAB Webinar 
% MATLAB Central File Exchange. Retrieved December 15, 2021.
% https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/63626/versions/1/previews/Demos/2_fuelEconomyRegress/dummytable.m/index.html
%[2]%Lasso or elastic net regularization for generalized linear models - 
% MATLAB lassoglm - MathWorks United Kingdom2. 
% https://uk.mathworks.com/help/stats/lassoglm.html (accessed Dec. 15, 2021).
%[3]%Using Bayesopt for TreeBagger Classification -’, MATLAB. 
% https://uk.mathworks.com/matlabcentral/answers/347547-using-bayesopt-for-treebagger-classification (accessed Dec. 15, 2021).


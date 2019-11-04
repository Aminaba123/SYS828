% ----------------------------
% MOHAMMADAMIN ABBASNEJAD
% ----------------------------
% SYS800 Pattern recognition
% Lab 2
% Report 2
% ----------------------------
% Experimentation of 3 different classifiers 
% kNN, quadratic bayes and SVM
% ----------------------------
clear all; close all; clc;
addpath(genpath('LibrairieSVM\public'));


% Load data
% Profile
% Retine
load('Grille_features_95.mat');

%% Definition of global variables
trainbase = Train_features; % adapt variable names
testbase = Test_features;
nbClasses = length(unique(test_labels)); % number of classes

% Shuffling of the test dataset
newordertestbase = randperm(size(testbase, 1));
newtestbase = testbase(newordertestbase, :);
newtestlabels = test_labels(newordertestbase);

%% V) Classification with quadratic bayes method
disp('-------------------------------------------');
disp('Classification with quadractic bayes method');
disp('-------------------------------------------');
tic;
disp('Learning of the training database characteristics');
[matcov, matmu] = modelQuadBayes(trainbase, train_labels);
tTrain = toc;
disp(['training time : ', num2str(tTrain), ' s']);

%% Determination of the accuracy of the learned model with the testbase dataset
disp('Determination of the accuracy of the learned model with the test dataset');
tic;
% Classification of the test samples
quadbayeslabels = classQuadBayes(matcov, matmu, newtestbase);
quadbayesaccuracy = accuracy(quadbayeslabels, newtestlabels);
disp(['test accuracy : ', num2str(quadbayesaccuracy)]);
tTest = toc;
disp(['testing time : ' num2str(tTest) 's']);

%% Generation of the confusion matrix
disp('Confusion matrix: ');
matConf = confMat(quadbayeslabels, newtestlabels)
matMisClassified = [];
% Generation of a table of misclassified examples
for idxMisClassified = 1:length(quadbayeslabels)
    if quadbayeslabels(idxMisClassified) ~= newtestlabels(idxMisClassified)
        matMisClassified = [matMisClassified; [newordertestbase(idxMisClassified), newtestlabels(idxMisClassified), quadbayeslabels(idxMisClassified)]];
    end
end
save('missclassified_Quad_Bayes', 'matMisClassified');

%% VI) Classification with KNN method
disp('------------------------------');
disp('Classification with KNN method');
disp('------------------------------');

% Determination of best k with hold out method
disp('Determination of best k with hold out method');
tic;
knnacc = zeros(1, 10);
for idxk = 1:10
    % Shuffling of the samples of the training database
    newordertrainbase = randperm(size(trainbase, 1));
    newtrainbase = trainbase(newordertrainbase, :);
    newtrainlabels = train_labels(newordertrainbase);
    % Separation of the training dataset in training and validation dataset
    knnlabels = classKNN(idxk, newtrainbase(1:4000,:), newtrainlabels(1:4000), newtrainbase(4001:end,:));
    % Calculation of the accuracy
    knnacc(idxk) = accuracy(knnlabels, newtrainlabels(4001:end));
end
tTrain = toc;
[bestacc, bestk] = max(knnacc);
disp(['best k: ' num2str(bestk)]);
disp(['best accuracy: ' num2str(bestacc)]);
disp(['training time : ', num2str(tTrain), ' s']);

figure;
plot(knnacc)
title('Variation of the accuracy in function of K');
xlabel('K');
ylabel('Accuracy');

%% Determination of the accuracy of the learned model with the test dataset
disp('Determination of the accuracy of the learned model with the test dataset');
tic;
% Test on test dataset with best K
knnlabels = classKNN(bestk, trainbase, train_labels, newtestbase);
knnaccuracy = accuracy(knnlabels, newtestlabels);
disp(['test accuracy : ', num2str(knnaccuracy)]);
tTest = toc;
disp(['testing time : ' num2str(tTest) 's']);

%% Generation of the confusion matrix
disp('Confusion matrix: ');
matConf = confMat(knnlabels, newtestlabels)
% Generation of a table of misclassified examples
matMisClassified = [];
for idxMisClassified = 1:length(knnlabels)
    if knnlabels(idxMisClassified) ~= newtestlabels(idxMisClassified)
        matMisClassified = [matMisClassified; [newordertestbase(idxMisClassified), newtestlabels(idxMisClassified), knnlabels(idxMisClassified)]];
    end
end
save('missclassified_KNN', 'matMisClassified');

%% VII) Classification with SVM method
disp('------------------------------');
disp('Classification with SVM method');
disp('------------------------------');
disp('Determination of best c and gamma parameters');
tic;
[bestC, bestGamma] = modelSVM(trainbase, train_labels);
tTrain = toc;
disp(['training time : ', num2str(tTrain), ' s']);

%% Determination of the accuracy on the test dataset with the model
% with the best parameters c and gamma
disp('Determination of the accuracy of the learned model with the test dataset');
tic;
svmlabels = classSVM(trainbase, trainlabels, newtestbase, newtestlabels, bestC, bestGamma);
svmaccuracy = accuracy(svmlabels, newtestlabels);
disp(['test accuracy : ', num2str(svmaccuracy)]);
tTest = toc;
disp(['testing time : ' num2str(tTest) 's']);

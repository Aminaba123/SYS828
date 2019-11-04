% Mohammadamin Abbasnejad
% Lab 3
% -------------------------------------------------------------------------
% Face classification by the kNN method using cross-validation (5 fold)
% technique using PCA
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);

%% 
nbTotalSubjects = 40;
nbTotalImages = 10;
nbSubjects = 40;
disp(['Nombre of subjects : ' num2str(nbSubjects)])
nbImagesTrain = 5;
disp(['Number of images per subject in training set : ' num2str(nbImagesTrain)])
nbImageTest = nbTotalImages - nbImagesTrain;
disp(['Number of images per subject in test set: ' num2str(nbImageTest)])

nReplications = 5;
errClassifier = zeros(1,nReplications);

for r = 1:nReplications
    
    disp(['REP ' num2str(r)])
    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);        
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImagesTrain));
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImagesTrain+1:end));
    Weight_pca = pcam(training_dataset);
    nbMaxFeatures = size(Weight_pca, 2);
    training_dataset_pca = training_dataset * Weight_pca;
    test_dataset_pca = test_dataset * Weight_pca;
    tic;
    classifier =  knnc([], 1);
    nFolds = 5;
    logError_nbFeatures = zeros(1, nbMaxFeatures);   
    for nbFeatures = 1:nbMaxFeatures
        mixmat = randperm(size(training_dataset_pca, 1));
        error = prcrossval(training_dataset_pca(mixmat, 1:nbFeatures), classifier, nFolds);
        logError_nbFeatures(nbFeatures) = error;
    end
    tTrain = toc;
    disp(['Training time: ' num2str(tTrain) 's']);       
   
    [bestError_nbFeatures, bestNbFeatures] = min(logError_nbFeatures);
    disp(['The optimal number of features is: ' num2str(bestNbFeatures)]);
    disp(['Validation error for 1nn with this number of features: ' num2str(bestError_nbFeatures*100) '%']);
    figure(100+r)
    plot(logError_nbFeatures)
    xlabel('Number of features')
    title(['Error based on the number of features for repetition'...
    num2str(r)])   
    tic;
    nFolds = 5;
    logError_k = zeros(1, 10);
    for k = 1:10
        classifier =  knnc([], k); 
        error = prcrossval(training_dataset_pca(:, 1:bestNbFeatures), classifier, nFolds);
        logError_k(k) = error;
    end
    tTrain = toc;
    disp(['Training time: ' num2str(tTrain) 's']);       

    [bestError_k, bestK] = min(logError_k);
    disp(['The best k is: ' num2str(bestK)]);
    disp(['Validation Err: ' num2str(bestError_k*100) '%']);
    figure(1000+r)
    plot(logError_k)
    xlabel('k')
    title(['Error based on the number of neighbors for the repeat'...
    num2str(r)]) 
    tic;
    classifier =  knnc(training_dataset_pca, bestK); 
    errClassifier(r) = testc(test_dataset_pca, classifier);
    tTest = toc;
    disp(['Test time: ' num2str(tTest) 's']);        
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset_pca), labeld(test_dataset_pca * classifier))
    listMisclassified = [getlabels(test_dataset_pca), labeld(test_dataset_pca * classifier)];
    maskMisclassified = listMisclassified(:, 1) ~= listMisclassified(:, 2);
    matMissclassified = test_dataset(maskMisclassified, :);
    listMisclassified = listMisclassified(maskMisclassified, :);
    for idxImage = 1:size(listMisclassified, 1)
        image = uint8(reshape(+matMissclassified(idxImage, :), 112, 92));
        imwrite(image, sprintf(['knn_pca_' num2str(r) '_' num2str(listMisclassified(idxImage, 1)) '_' num2str(listMisclassified(idxImage, 2)) '.png']))
    end
end
save('matconf_knn_pca', 'confMat');
disp('Classification Performance')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
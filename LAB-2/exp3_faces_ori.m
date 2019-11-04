% Mohammadamin Abbasnejad
% Lab 3
% -------------------------------------------------------------------------
% Face classification by the kNN method using cross-validation (5 fold) technique
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
nbTrainImage = 5;
disp(['Number of images per subject in training set: ' num2str(nbTrainImage)])
nbTestImage = nbTotalImages - nbTrainImage;
disp(['Number of images per subject in test set: ' num2str(nbTestImage)])

nReplications = 5;
errClassifier = zeros(1,nReplications);

for r = 1:nReplications
    
    disp(['REP ' num2str(r)])

    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbTrainImage));
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbTrainImage+1:end));
    tic;
    nFolds = 5;
    logError_k = zeros(1, 10);
    for k = 1:10
        classifier =  knnc([], k);
        error = prcrossval(training_dataset, classifier, nFolds);
        logError_k(k) = error;
    end
    tTrain = toc;
    disp(['Training time: ' num2str(tTrain) 's']);
    [bestError_k, bestK] = min(logError_k);
    disp(['Best K is: ' num2str(bestK)]);
    disp(['Validation Err: ' num2str(bestError_k*100) '%']);
    figure(1000+r)
    plot(logError_k)
    xlabel('k')
    title(['Error based on the number of neighbors for the repeat '...
    num2str(r)]) 
    tic;
    classifier =  knnc(training_dataset, bestK); 
    errClassifier(r) = testc(test_dataset, classifier);
    tTest = toc;
    disp(['Testing time: ' num2str(tTest) 's']);        
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset), labeld(test_dataset * classifier));
    listMisclassified = [getlabels(test_dataset), labeld(test_dataset * classifier)];
    maskMisclassified = listMisclassified(:, 1) ~= listMisclassified(:, 2);
    matMissclassified = test_dataset(maskMisclassified, :);
    listMisclassified = listMisclassified(maskMisclassified, :);
    for idxImage = 1:size(listMisclassified, 1)
        image = uint8(reshape(+matMissclassified(idxImage, :), 112, 92));
        imwrite(image, sprintf(['knn_ori_' num2str(r) '_' num2str(listMisclassified(idxImage, 1)) '_' num2str(listMisclassified(idxImage, 2)) '.png']))
    end
end
save('matconf_knn_ori', 'confMat');
disp('Classification Performance')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
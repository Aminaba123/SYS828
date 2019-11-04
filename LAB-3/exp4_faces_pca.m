% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Face classification by the SVM method after PCA using cross-validation
% technique (5 fold)
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);
nbTotalSubjects = 40;
nbTotalImages = 10;
nbSubjects = 40;
disp(['Nombre of subjects: ' num2str(nbSubjects)])
nbImageTrain = 5;
disp(['Number of images subject in the training base: ' num2str(nbImageTrain)])
nbImageTest = nbTotalImages - nbImageTrain;
disp(['Number of images per subject in the test database: ' num2str(nbImageTest)])


%% 
nReplications = 5;
errClassifier = zeros(1, nReplications);

for r = 1:nReplications
    
    disp(['REP ' num2str(r)])
    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);
        
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    
    Weight_pca = pcam(training_dataset);
    training_dataset_pca = training_dataset * Weight_pca;
    test_dataset_pca = test_dataset * Weight_pca;
    min_dataset = ones(size(training_dataset_pca)) .* min(training_dataset_pca);
    max_dataset = ones(size(training_dataset_pca)) .* max(training_dataset_pca);
    training_dataset_pca = (training_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
    min_dataset = ones(size(test_dataset_pca)) .* min(test_dataset_pca);
    max_dataset = ones(size(test_dataset_pca)) .* max(test_dataset_pca);
    test_dataset_pca = (test_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
        tic;
    nFolds = 5; 
    gamma = [0.1 1 10];
    c = [0.1 1 10 100];
    logError = nan(length(gamma), length(c)); 
    for idxGamma = 1:length(gamma)
        for idxC = 1:length(c)
        
        classifier =  svc([], proxm('r', gamma(idxGamma)), c(idxC));
        
        mixmat = randperm(size(training_dataset_pca, 1));
        
        logError(idxGamma, idxC) = prcrossval(training_dataset_pca(mixmat, 1:35), classifier, nFolds);
        end
    end
    tTrain = toc;
    disp(['Training time: ' num2str(tTrain) 's']);     

    figure(1000+r)
    mesh(logError)
    xlabel('index of C')
    ylabel('index of gamma')
    title(['Error based on gamma and c parameters for repetition'...
        num2str(r)])

    [bestError, idxBestError] = min(logError(:));
    [idxGamma, idxC] = ind2sub(size(logError), idxBestError);
    bestGamma = gamma(idxGamma);
    bestC = c(idxC);
    disp(['The best gamma is: ' num2str(bestGamma)]);
    disp(['The best c is : ' num2str(bestC)]);
    disp(['Validation Err: ' num2str(bestError*100) '%']);
    tic;
    mixmat = randperm(size(training_dataset_pca, 1));
    classifier =  svc(training_dataset_pca(mixmat, 1:35), proxm('r', bestGamma), bestC);
    mixmat = randperm(size(test_dataset_pca, 1));
    errClassifier(r) = testc(test_dataset_pca(mixmat, 1:35), classifier);
    tTest = toc;
    disp(['Test time: ' num2str(tTest) 's']);  
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset_pca(mixmat, :)), labeld(test_dataset_pca(mixmat, 1:35) * classifier))
    listMisclassified = [getlabels(test_dataset_pca(mixmat, :)), labeld(test_dataset_pca(mixmat, 1:35) * classifier)];
    maskMisclassified = listMisclassified(:, 1) ~= listMisclassified(:, 2);
    matMissclassified = (test_dataset(mixmat, :));
    matMissclassified = matMissclassified(maskMisclassified, :);
    listMisclassified = listMisclassified(maskMisclassified, :);
    for idxImage = 1:size(listMisclassified, 1)
        image = uint8(reshape(+matMissclassified(idxImage, :), 112, 92));
        imwrite(image, sprintf(['svm_pca_' num2str(r) '_' num2str(listMisclassified(idxImage, 1)) '_' num2str(listMisclassified(idxImage, 2)) '.png']))
    end
end
save('matconf_svm_pca', 'confMat');
disp('Classification Performance')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
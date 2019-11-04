% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Classification of faces by vote of 2 classifiers SVM and
% 1 KNN classifier after PCA
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

nReplications = 5;
errClassifier = zeros(1, nReplications);

for r = 1:nReplications
    disp(['REP ' num2str(r)])
    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);
        

    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));

    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    
    
    W_pca = pcam(training_dataset);
    training_dataset_pca = training_dataset * W_pca;
    test_dataset_pca = test_dataset * W_pca;

    
    min_dataset = ones(size(training_dataset_pca)) .* min(training_dataset_pca);
    max_dataset = ones(size(training_dataset_pca)) .* max(training_dataset_pca);
    training_dataset_pca = (training_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
    min_dataset = ones(size(test_dataset_pca)) .* min(test_dataset_pca);
    max_dataset = ones(size(test_dataset_pca)) .* max(test_dataset_pca);
    test_dataset_pca = (test_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
    


    %% 
    tic;
    mixmat = randperm(size(training_dataset_pca, 1));
    svm1 =  svc(training_dataset_pca(mixmat, 1:35), proxm('r', 1), 1);
    svm2 =  svc(training_dataset_pca(mixmat, 1:35), proxm('r', 1), 10);
    knn1 =  knnc(training_dataset_pca(mixmat, 1:35), 1); 
    mixmat = randperm(size(test_dataset_pca, 1));
    
    myCbn = [knn1, svm1, svm2]*votec;
    errClassifier(r) = testc(test_dataset_pca(mixmat, 1:35), myCbn);
    tTest = toc;
    disp(['Test time: ' num2str(tTest) 's']);      
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset_pca(mixmat, :)), labeld(test_dataset_pca(mixmat, 1:35) * myCbn));    
end

save('comb_pca_matconf', 'confMat');

%% Classification
disp('==============================================')
disp('FINAL PERFORMANCE OF THE CLASSIFIER')
disp('----------------------------------------------')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
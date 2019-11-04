% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Classification of faces by 2 classifiers SVM and
% KNN after LDA
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
accClassifier = zeros(1, nReplications);
%% 
nReplications = 5;
errClassifier = zeros(1, nReplications);

for r = 1:nReplications
    
   
    disp(['REP ' num2str(r)])
    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);
        
    
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));
    
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    Weight_lda = fisherm(training_dataset);
    training_dataset_lda = training_dataset * Weight_lda;
    test_dataset_lda = test_dataset * Weight_lda;
  
    tic;
    mixmat = randperm(size(training_dataset_lda, 1));
    svm1 =  svc(training_dataset_lda(mixmat, :), proxm('r', 10^23), 10^12);
    knn1 =  knnc(training_dataset_lda(mixmat, :), 1); 
  
    myCbn = [knn1, svm1]*votec;
    mixmat = randperm(size(test_dataset_lda, 1));
    errClassifier(r) = testc(test_dataset_lda(mixmat, :), myCbn);
    tTest = toc;
    disp(['Temps test : ' num2str(tTest) 's']);       
    disp(['Erreur test : ' num2str(errClassifier(r)*100) '%']);
    disp('Matrice de confusion: ');
    confMat{r} = confmat(getlabels(test_dataset_lda(mixmat, :)), labeld(test_dataset_lda(mixmat, :) * myCbn));
end

save('comb_lda_matconf', 'confMat');

disp('Classification Performance')
disp('----------------------------------------------')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
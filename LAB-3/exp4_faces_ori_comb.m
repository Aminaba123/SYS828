% -------------------------------------------------------------------------
% Julie Lecolier
% Lucas Shorten
% Saypraseuth Mounsaveng
% -------------------------------------------------------------------------
% SYS828 Systemes biometriques
% Automne 2017
% Lab 4
% -------------------------------------------------------------------------
% Classification de visages par vote de 2 classificateurs SVM et 
% 1 classificateur KNN apres application de PCA
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);

% Initialisation des variables globales
nbTotalPersons = 40;
nbTotalImages = 10;
nbPersons = 40;
disp(['Nombre of subjects: ' num2str(nbPersons)])
nbImageTrain = 5;
disp(['Number of images subject in the training base: ' num2str(nbImageTrain)])
nbImageTest = nbTotalImages - nbImageTrain;
disp(['Number of images per subject in the test database: ' num2str(nbImageTest)])


%% Deroulement des replications
nReplications = 5;
errClassifier = zeros(1, nReplications);

for r = 1:nReplications
    
    disp('==============================================')
    disp(['REP ' num2str(r)])
    disp('----------------------------------------------')

    % Melange des personnes seulement si on selectionne toutes les
    % personnes du dataset, sinon prcrossval a des problemes avec des
    % labels de valeur superieures au nombre de classes
    % newOrderPersons = randperm(nbTotalPersons, nbPersons);
    newOrderPersons = 1:nbPersons;
    % Melange des photos
    newOrderFaces = randperm(nbTotalImages);
        
    %% Lecture des images
    % Repartition du dataset en 2 sous ensembles training / testing
    % Training dataset
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));
    % Test dataset
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    
    % Normalisation des donnees
    min_dataset = ones(size(training_dataset)) .* min(training_dataset);
    max_dataset = ones(size(training_dataset)) .* max(training_dataset);
    training_dataset = (training_dataset - min_dataset) ./ (max_dataset-min_dataset);
    min_dataset = ones(size(test_dataset)) .* min(test_dataset);
    max_dataset = ones(size(test_dataset)) .* max(test_dataset);
    test_dataset = (test_dataset - min_dataset) ./ (max_dataset-min_dataset);
    
    %% Evaluation de la performance sur la base de test.
    tic;
    mixmat = randperm(size(training_dataset, 1));
    svm1 =  svc(training_dataset(mixmat, :), proxm('r', 10), 10);
    svm2 =  svc(training_dataset(mixmat, :), proxm('r', 10), 100);
    knn1 =  knnc(training_dataset(mixmat, :), 1); 
    mixmat = randperm(size(test_dataset, 1));
    % Combine the classifiers with some combination rule (e.g. product rule)
    myCbn = [knn1, svm1, svm2]*votec;
    errClassifier(r) = testc(test_dataset(mixmat, :), myCbn);
    tTest = toc;
    disp(['Test time: ' num2str(tTest) 's']);      
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset(mixmat, :)), labeld(test_dataset(mixmat, :) * myCbn));    
end
% Sauvegarde de la matrice de confusion pour exploitation ulterieure
save('comb_ori_matconf', 'confMat');

%% Compilation des résultats
disp('==============================================')
disp('FINAL PERFORMANCE OF THE CLASSIFIER')
disp('----------------------------------------------')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
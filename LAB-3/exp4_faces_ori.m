% -------------------------------------------------------------------------
% Julie Lecolier
% Lucas Shorten
% Saypraseuth Mounsaveng
% -------------------------------------------------------------------------
% SYS828 Systemes biometriques
% Automne 2017
% Lab 4
% -------------------------------------------------------------------------
% Classification de visages par la methode SVM en utilisant la technique
% de cross-validation et 5 replications et sans reduction de dimension
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
    training_dataset_ori = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));
    % Test dataset
    test_dataset_ori = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    
    training_dataset = training_dataset_ori;
    test_dataset = test_dataset_ori;
    
    % Normalisation des donnees
    min_dataset = ones(size(training_dataset)) .* min(training_dataset);
    max_dataset = ones(size(training_dataset)) .* max(training_dataset);
    training_dataset = (training_dataset - min_dataset) ./ (max_dataset-min_dataset);
    min_dataset = ones(size(test_dataset)) .* min(test_dataset);
    max_dataset = ones(size(test_dataset)) .* max(test_dataset);
    test_dataset = (test_dataset - min_dataset) ./ (max_dataset-min_dataset);
    
    %% Determination de la valeur optimale de c et gamma en utilisant la cross-validation
    disp('**********************************************************************************');
    disp('Determination of the optimal value of the c and gamma parameters by cross-validation');
    disp('**********************************************************************************');
    tic;
    nFolds = 5; % Nombre de blocs pour la validation croisee
    gamma = [0.1 1 10 100];
    c = [1 10 100 1000];
    logError = nan(length(gamma), length(c)); 
    for idxGamma = 1:length(gamma)
        for idxC = 1:length(c)
        % Initialisation du classificateur
        classifier =  svc([], proxm('r', gamma(idxGamma)), c(idxC));
        % Melange des echantillons du training dataset
        mixmat = randperm(size(training_dataset, 1));
        % Validation croisee avec nFolds
        logError(idxGamma, idxC) = prcrossval(training_dataset(mixmat, :), classifier, nFolds);
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
       
    %% Affichage de la valeur k optimale
    [bestError, idxBestError] = min(logError(:));
    [idxGamma, idxC] = ind2sub(size(logError), idxBestError);
    bestGamma = gamma(idxGamma);
    bestC = c(idxC);
    disp(['The best gamma is: ' num2str(bestGamma)]);
    disp(['The best c is : ' num2str(bestC)]);
    disp(['Validation Err: ' num2str(bestError*100) '%']);

    %% Evaluation de la performance sur la base de test.
    tic;
    mixmat = randperm(size(training_dataset, 1));
    classifier =  svc(training_dataset(mixmat, :), proxm('r', bestGamma), bestC);
    mixmat = randperm(size(test_dataset, 1));
    errClassifier(r) = testc(test_dataset(mixmat, :), classifier);
    tTest = toc;
    disp(['Test time: ' num2str(tTest) 's']);  
    disp(['Test Err: ' num2str(errClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat{r} = confmat(getlabels(test_dataset), labeld(test_dataset(mixmat, :) * classifier))
    
    % Extraction of the misclassified samples
    listMisclassified = [getlabels(test_dataset(mixmat, :)), labeld(test_dataset(mixmat, :) * classifier)];
    maskMisclassified = listMisclassified(:, 1) ~= listMisclassified(:, 2);
    matMissclassified = (test_dataset_ori(mixmat, :));
    matMissclassified = matMissclassified(maskMisclassified, :);
    listMisclassified = listMisclassified(maskMisclassified, :);
    for idxImage = 1:size(listMisclassified, 1)
        image = uint8(reshape(+matMissclassified(idxImage, :), 112, 92));
        imwrite(image, sprintf(['svm_ori_' num2str(r) '_' num2str(listMisclassified(idxImage, 1)) '_' num2str(listMisclassified(idxImage, 2)) '.png']))
    end
end
% Sauvegarde de la matrice de confusion pour exploitation ulterieure
save('svm_ori_matconf', 'confMat');


%% Compilation des résultats
disp('==============================================')
disp('FINAL PERFORMANCE OF THE CLASSIFIER')
disp('----------------------------------------------')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
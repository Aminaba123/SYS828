% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Face classification by the SVM method after LDA 
% using the cross-validation technique and 5 replications
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
     
    %% 
    tic;
    nFolds = 5;
    gamma = [10^23 10^24];
    c = [10^12 10^13 10^14];
    logError = nan(length(gamma), length(c)); 
    for idxGamma = 1:length(gamma)
        for idxC = 1:length(c)
    
        classifier =  svc([], proxm('r', gamma(idxGamma)), c(idxC));
    
        mixmat = randperm(size(training_dataset_lda, 1));
    
        logError(idxGamma, idxC) = prcrossval(training_dataset_lda(mixmat,:), classifier, nFolds);
        end
    end
    tTrain = toc;
    disp(['Temps entrainement : ' num2str(tTrain) 's']);    
    
    figure(1000+r)
    mesh(logError)
    xlabel('index of C')
    ylabel('index of gamma')
    title(['Erreur en fonction des parametres gamma et c pour la répétition '...
        num2str(r)])
    [bestError, idxBestError] = min(logError(:));
    [idxGamma, idxC] = ind2sub(size(logError), idxBestError);
    bestGamma = gamma(idxGamma);
    bestC = c(idxC);
    disp(['Le meilleur gamma est : ' num2str(bestGamma)]);
    disp(['Le meilleur c est : ' num2str(bestC)]);
    disp(['Erreur validation : ' num2str(bestError*100) '%']);
    tic;
    mixmat = randperm(size(training_dataset_lda, 1));
    classifier =  svc(training_dataset_lda(mixmat, :), proxm('r', bestGamma), bestC);
    mixmat = randperm(size(test_dataset_lda, 1));
    errClassifier(r) = testc(test_dataset_lda(mixmat, :), classifier);
    tTest = toc;
    disp(['Temps test : ' num2str(tTest) 's']);    
    disp(['Erreur test : ' num2str(errClassifier(r)*100) '%']);
    disp('Matrice de confusion: ');
    confMat{r} = confmat(labeld(test_dataset_lda(mixmat, :) * classifier), getlabels(test_dataset_lda(mixmat, :)))
    
    
    listMisclassified = [getlabels(test_dataset_lda(mixmat, :)), labeld(test_dataset_lda(mixmat, :) * classifier)];
    maskMisclassified = listMisclassified(:, 1) ~= listMisclassified(:, 2);
    matMissclassified = (test_dataset(mixmat, :));
    matMissclassified = matMissclassified(maskMisclassified, :);
    listMisclassified = listMisclassified(maskMisclassified, :);
    for idxImage = 1:size(listMisclassified, 1)
        image = uint8(reshape(+matMissclassified(idxImage, :), 112, 92));
        imwrite(image, sprintf(['svm_lda_' num2str(r) '_' num2str(listMisclassified(idxImage, 1)) '_' num2str(listMisclassified(idxImage, 2)) '.png']))
    end
end

save('matconf_svm_lda', 'confMat');

disp('Classification Performance')
disp(['Classification rate: ' num2str((1-mean(errClassifier))*100) '('...
    num2str(std(errClassifier)*100) ') %'])
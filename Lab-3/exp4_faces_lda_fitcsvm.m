% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Face classification by the SVM method after PCA
% using cross-validation (5 fold)
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);

nbTotalSubjects = 40;
nbTotalImages = 10;
nbSubjects = 40;
disp(['Nombre de personnes : ' num2str(nbSubjects)])
nbImageTrain = 5;
disp(['Nombre d images par personne dans la base d entrainement : ' num2str(nbImageTrain)])
nbImageTest = nbTotalImages - nbImageTrain;
disp(['Nombre d images par personne dans la base de test : ' num2str(nbImageTest)])

%% 
nReplications = 5;
accClassifier = zeros(1, nReplications);

for r = 1:nReplications
    
    disp(['REP ' num2str(r)])
    
    newOrderPersons = 1:nbSubjects;
    
    newOrderFaces = randperm(nbTotalImages);
        
    
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));
    
    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    
    Weight_lda = fisherm(training_dataset);
    training_dataset_lda = training_dataset * Weight_lda;
    test_dataset_lda = test_dataset * Weight_lda;

    training_labels_pca = getlabels(training_dataset_lda)';
    training_dataset_lda = +training_dataset_lda;

    test_labels_pca = getlabels(test_dataset_lda)';
    test_dataset_lda = +test_dataset_lda;
    tic;
    nFolds = 5; 
    gamma = [10^23 10^24];
    c = [10^12 10^13 10^14];
    nbClasses = nbSubjects;
    logAcc = nan(length(gamma), length(c)); 
    for idxGamma = 1:length(gamma)
        for idxC = 1:length(c)
            nbTestSamples = nbSubjects * nbImageTrain;
            
            nbValSamples = round(nbTestSamples / nFolds);
            
            neworder = randperm(size(training_dataset_lda, 1));
            newtrainbase = training_dataset_lda(neworder, :);
            newtrainlabels = training_labels_pca(neworder);
            matscorefold = zeros(1,nFolds);
            for idxFold = 1:nFolds
            
                newvalbase = newtrainbase((end - nbValSamples + 1:end), :);
                newvallabels = newtrainlabels((end - nbValSamples + 1:end));
                newtrainbase = newtrainbase((1:end - nbValSamples), :);
                newtrainlabels = newtrainlabels((1:end - nbValSamples));
            
                matclass = zeros(size(newvalbase, 1), nbClasses);
                for idxClass = 1:nbClasses
            
                    newtrainlabelslog = newtrainlabels;
                    newtrainlabelslog(newtrainlabels ~= idxClass) = -1;
                    newtrainlabelslog(newtrainlabels == idxClass) = 1;
            
                    if sum(newtrainlabelslog == 1) ~= 0
                        classifier = fitcsvm(newtrainbase, newtrainlabelslog, 'KernelFunction', 'gaussian', 'KernelScale', gamma(idxGamma), 'BoxConstraint' , c(idxC));
                    else
            
                        classifier = fitcsvm(newtrainbase, newtrainlabelslog, 'KernelFunction', 'gaussian', 'KernelScale', gamma(idxGamma), 'BoxConstraint' , 1);
                    end
            
                    newvallabelslog = newvallabels;
                    newvallabelslog(newvallabels ~= idxClass) = -1;
                    newvallabelslog(newvallabels == idxClass) = 1;    
                    [predictedLabels, score] = predict(classifier, newvalbase);
                    matclass(:, idxClass) = (predictedLabels) .* score(:, 1);
                end
            
                [~, svmlabels] = max(matclass, [], 2);
            
                svmaccuracy = sum(svmlabels' == newvallabels) / length(svmlabels');
                matscorefold(idxFold) = svmaccuracy;
            
            
            end
            logAcc(idxGamma, idxC) = mean(matscorefold);
        end
    end
    tTrain = toc;
    disp(['Temps entrainement : ' num2str(tTrain) 's']); 

    figure(1000+r)
    mesh(logAcc)
    xlabel('index of C')
    ylabel('index of gamma')
    title(['Precision en fonction des parametres gamma et c pour la répétition '...
        num2str(r)])
       
    
    [bestAcc, idxBestAcc] = max(logAcc(:));
    [idxGamma, idxC] = ind2sub(size(logAcc), idxBestAcc);
    bestGamma = gamma(idxGamma);
    bestC = c(idxC);
    disp(['Le meilleur gamma est : ' num2str(bestGamma)]);
    disp(['Le meilleur c est : ' num2str(bestC)]);
    disp(['Accuracy validation : ' num2str(bestAcc*100) '%']);

    
    tic;
    nbTestSamples = nbSubjects * nbImageTrain;
    
    neworder = randperm(nbTestSamples);
    newordertest = randperm(length(test_labels_pca));
    
    newtrainbase = training_dataset_lda(neworder, :);
    newtrainlabels = training_labels_pca(neworder);
    newtestbase = test_dataset_lda(newordertest, :);
    newtestlabels = test_labels_pca(newordertest);
    
    matclass = zeros(size(newtrainbase, 1), nbClasses);
    for idxClass = 1:nbClasses
    
        newtrainlabelslog = newtrainlabels;
        newtrainlabelslog(newtrainlabels ~= idxClass) = -1;
        newtrainlabelslog(newtrainlabels == idxClass) = 1;
    
        classifier = fitcsvm(newtrainbase, newtrainlabelslog, 'KernelFunction', 'gaussian', 'KernelScale', bestGamma, 'BoxConstraint' , bestC);
    
        newtestlabelslog = newtestlabels;
        newtestlabelslog(newtestlabels ~= idxClass) = -1;
        newtestlabelslog(newtestlabels == idxClass) = 1;    
        [predictedLabels, score] = predict(classifier, newtestbase);
        matclass(:, idxClass) = (predictedLabels) .* score(:, 1);
    end
    
    [~, svmlabels] = max(matclass, [], 2);
    tTest = toc;
    disp(['Temps test : ' num2str(tTest) 's']);        
    
    accClassifier(r) = sum(svmlabels' == newtestlabels) / length(svmlabels');
    disp(['Accuracy test : ' num2str(accClassifier(r)*100) '%']);
    disp('Matrice de confusion: ');
    confMat = confmat(newtestlabels', svmlabels)
end


disp('Classification Performance')
disp('----------------------------------------------')
disp(['Classification rate: ' num2str((mean(accClassifier))*100) '('...
    num2str(std(accClassifier)*100) ') %'])
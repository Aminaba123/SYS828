% Mohammadamin Abbasnejad
% Lab 4
% -------------------------------------------------------------------------
% Face classification by the SVM method after PCA 
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

for r = 1:nReplications
    
    disp(['REP ' num2str(r)])
    newOrderPersons = 1:nbSubjects;
    newOrderFaces = randperm(nbTotalImages);
    training_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(1:nbImageTrain));

    test_dataset = read_faces('att_faces/', newOrderPersons, newOrderFaces(nbImageTrain+1:end));
    

    W_pca = pcam(training_dataset);
    training_dataset_pca = training_dataset * W_pca;
    test_dataset_pca = test_dataset * W_pca;


    training_labels_pca = getlabels(training_dataset_pca)';
    training_dataset_pca = +training_dataset_pca;

    test_labels_pca = getlabels(test_dataset_pca)';
    test_dataset_pca = +test_dataset_pca;
    
    min_dataset = ones(size(training_dataset_pca)) .* min(training_dataset_pca);
    max_dataset = ones(size(training_dataset_pca)) .* max(training_dataset_pca);
    training_dataset_pca = (training_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
    min_dataset = ones(size(test_dataset_pca)) .* min(test_dataset_pca);
    max_dataset = ones(size(test_dataset_pca)) .* max(test_dataset_pca);
    test_dataset_pca = (test_dataset_pca - min_dataset) ./ (max_dataset-min_dataset);
    
    tic;    
    nFolds = 5; 
    gamma = [0.1 1 10 100];
    c = [10 100 1000 10000];
    nbClasses = nbSubjects;
    logAcc = nan(length(gamma), length(c)); 
    for idxGamma = 1:length(gamma)
        for idxC = 1:length(c)
            nbTestSamples = nbSubjects * nbImageTrain;
            
            nbValSamples = round(nbTestSamples / nFolds);
            
            neworder = randperm(size(training_dataset_pca, 1));
            newtrainbase = training_dataset_pca(neworder, 1:35);
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

                newtrainbase = [newvalbase; newtrainbase];
                newtrainlabels = [newvallabels, newtrainlabels];
            end
            logAcc(idxGamma, idxC) = mean(matscorefold);
        end
    end
    tTrain = toc;
    disp(['Training time: ' num2str(tTrain) 's']);  
    
    figure(1000+r)
    mesh(logAcc)
    xlabel('index of C')
    ylabel('index of gamma')
    title(['Precision according to gamma and c parameters for repetition'...
        num2str(r)])
       
    %% 
    [bestAcc, idxBestAcc] = max(logAcc(:));
    [idxGamma, idxC] = ind2sub(size(logAcc), idxBestAcc);
    bestGamma = gamma(idxGamma);
    bestC = c(idxC);    
    disp(['The best gamma is: ' num2str(bestGamma)]);
    disp(['The best c  : ' num2str(bestC)]);
    disp(['Accuracy validation : ' num2str(bestAcc*100) '%']);

    %% Evaluation
    tic;
    nbTestSamples = nbSubjects * nbImageTrain;
    
    neworder = randperm(nbTestSamples);
    newordertest = randperm(length(test_labels_pca));

    newtrainbase = training_dataset_pca(neworder, 1:35);
    newtrainlabels = training_labels_pca(neworder);
    newtestbase = test_dataset_pca(newordertest, 1:35);
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
    disp(['Test timinh: ' num2str(tTest) 's']);      
    accClassifier(r) = sum(svmlabels' == newtestlabels) / length(svmlabels');
    disp(['Accuracy test : ' num2str(accClassifier(r)*100) '%']);
    disp('Confusion Matrix: ');
    confMat = confmat(newtestlabels', svmlabels)
end

%% Classification

disp('Classification Performance')

disp(['Classification rate: ' num2str((mean(accClassifier))*100) '('...
    num2str(std(accClassifier)*100) ') %'])
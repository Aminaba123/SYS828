% ----------------------------
% HoMOHAMMADAMIN ABBASNEJAD
% ----------------------------
% SYS800 Pattern recognition
% Lab 2
% Report 2
% ----------------------------
% Calculate the best c and gamma parameters of a SVM model 
% dataset
% Input
% trainbase : training samples (one row is one sample, one column a feature
% trainlabels : labels of the training samples (row vector)
% Output
% bestc : best c parameter for the given training dataset
% bestGamma : best gamma parameter for the given training dataset
% maxacc : validation accuracy obtained with best c and best gamma
% parameters
% ----------------------------
function [bestC, bestGamma] = modelSVM(trainbase, trainlabels)
% Determination of the number of classes
nbClasses = length(unique(trainlabels));

%% Shuffling of the dataset
newordertrainbase = randperm(size(trainbase, 1));
% Mixing of the samples in the training database
newtrainbase = trainbase(newordertrainbase(1:4000), :);
newtrainlabels = trainlabels(newordertrainbase(1:4000));
% Mixing of the samples in the validation database
newvalbase = trainbase(newordertrainbase(4001:end), :);
newvallabels = trainlabels(newordertrainbase(4001:end));

%% Determination of best parameters C et gamma with hold out method
gamma = [10^-6 10^-5 10^-4 0.001 0.01 0.1 1];
c = [1 10 10 1000 10000];

matscore = zeros(length(c), length(gamma));
% Create kind of status bar
disp(repmat('*',[1 length(c) * length(gamma)]));
for idxc = 1:length(c)
    for idxgamma = 1:length(gamma)
        % As svmtrain can only do 2-classes classification, we adopt the 1 VS
        % All strategy and Train 1 classifier per class
        matclass = zeros(2000, nbClasses);
        for idxClass = 1:nbClasses
            % Modify the labels vector to have only 2 classes
            % Set negative elements to -1
            newtrainlabelslog = newtrainlabels;
            newtrainlabelslog(newtrainlabels ~= (idxClass - 1)) = -1;
            newtrainlabelslog(newtrainlabels == (idxClass - 1)) = 1;
            newvallabelslog = newvallabels;
            newvallabelslog(newvallabels ~= (idxClass - 1)) = -1;
            newvallabelslog(newvallabels == (idxClass - 1)) = 1;
            % Train the 1 vs All svm model
            [~, row] = classifysvm(newtrainbase, newtrainlabelslog', newvalbase, newvallabelslog', gamma(idxgamma), c(idxc), c(idxc)); 
            % row = round(row);
            % matclass(:, idxClass) = (row + 1) * (1 - error);
            matclass(:, idxClass) = row;
        end
            fprintf('+');
            % Determine the class probability for each test sample
            [~, svmlabels] = max(matclass, [], 2);
            svmlabels = (svmlabels - 1)';
            svmaccuracy = accuracy(svmlabels, newvallabels);
            matscore(idxc, idxgamma) = svmaccuracy;
    end
end
fprintf('\n');
% Best accuracy
[maxacc, idxmax] = max(matscore(:));
[idxC, idxGamma] = ind2sub(size(matscore), idxmax);
disp(['best accuracy: ' num2str(maxacc)]);
% Best c
bestC = c(idxC);
disp(['best c: ' num2str(bestC)]);
% Best sigma
bestGamma = gamma(idxGamma);
disp(['best gamma: ' num2str(bestGamma)]);

% Graph of the error in funtion of c and gamma
figure;
mesh(matscore)
xlabel('Index of C')
ylabel('Index of gamma')
title('Accuracy in function of c and gamma parameters')
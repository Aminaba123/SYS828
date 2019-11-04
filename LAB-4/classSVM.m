% ----------------------------
% MOHAMMADAMIN ABBASNEJAD
% ----------------------------
% SYS800 Pattern recognition
% Lab 2
% Report 2
% ----------------------------
% Classify a dataset using SVM method
% Input
% trainbase : training samples (one row is one sample, one column a feature
% trainlabels : labels of the training samples (row vector)
% testbase : samples to classify
% testlabels : target labels of testbase dataset
% c : svm model c parameter
% gamma : svm model gamma parameter
% Output
% classlabels : predicted labels of the valbase dataset
% ----------------------------
function classlabels = classSVM(trainbase, trainlabels, testbase, testlabels, c, gamma)
% Determination of the number of classes
nbClasses = length(unique(trainlabels));

% Shuffling of the samples in the training database
newordertrainbase = randperm(size(trainbase, 1));
newtrainbase = trainbase(newordertrainbase, :);
newtrainlabels = trainlabels(newordertrainbase);

disp('Training of the model');
% Training of the model with the best parameters
matclass = zeros(size(testbase, 1), nbClasses);
disp(repmat('*',[1 nbClasses]));
for idxClass = 1:nbClasses
    fprintf('+');
    newtrainlabelslog = newtrainlabels;
    newtrainlabelslog(newtrainlabels ~= (idxClass - 1)) = - 1;
    newtrainlabelslog(newtrainlabels == (idxClass - 1)) = 1;
    newtestlabelslog = testlabels;
    newtestlabelslog(testlabels ~= (idxClass - 1)) = - 1;
    newtestlabelslog(testlabels == (idxClass - 1)) = 1;  
    % Train the 1 vs All svm model
    [error, row] = classifysvm(newtrainbase, newtrainlabelslog', testbase, newtestlabelslog', gamma, c, c); 
    row = round(row);
    matclass(:, idxClass)= (row + 1) * (1 - error); 
end
fprintf('\n');
% Determine the class for each test sample
[~, classlabels] = max(matclass, [], 2);
classlabels = (classlabels - 1)';

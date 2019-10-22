% Mohammadamin Abbasnejad
% Lab 2
% -------------------------------------------------------------------------
% Saving face data after applying PCA and LDA
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);

%% 
nbSubjects = [5 10 20 30 40];
nbImages = [2 3 5];

% Training dataset
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        train_dataset{(nbSubjects(idxNbPersons)),(nbImages(idxImage))} = read_faces('att_faces/', 1:(nbSubjects(idxNbPersons)),1:(nbImages(idxImage)));
    end
end

% Test dataset
test_dataset = read_faces('att_faces/', 1:40, 6:10);

% for LDA
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        Weight_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))} = fisherm(train_dataset{(nbSubjects(idxNbPersons)),(nbImages(idxImage))});
    end
end


% for PCA
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        [Weight_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}, frac_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}] = pcam(train_dataset{(nbSubjects(idxNbPersons)),(nbImages(idxImage))});
    end
end



% for PCA
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        train_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))} = train_dataset{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}  * Weight_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))};
    end
end

% for LDA
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        train_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))} = train_dataset{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}  * Weight_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))};
    end
end

save('train_dataset', 'train*', 'W*', 'frac*');
save('test_dataset', 'test*');
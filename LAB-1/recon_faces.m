% Mohammadamin Abbasnejad
% Lab 2
% Reconstruction of faces after PCA and LDA
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);

%%
nbSubjects = [5 20 40]; 
nbImages = [2 5]; 
load('test_dataset.mat');
load('train_dataset.mat','W*');

% for PCA
figure('NumberTitle', 'off', 'Name', 'Examples of face reconstruction using PCA');
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        
        idxImagOrig = randi(size(test_dataset,1),1); 
        origImag = test_dataset(idxImagOrig,:);
        
        reshapedImagOrig = reshape(uint8(double(test_dataset(idxImagOrig,:)))',112,92);
        subplot(length(nbSubjects) * length(nbImages),length(nbImages) + 2,(((idxNbPersons-1)*2+idxImage)-1)*4+1);
        imshow(reshapedImagOrig);
        title(sprintf('pca,%d,%d,orig',[nbSubjects(idxNbPersons), nbImages(idxImage)]));
        
        test_dataset_proj = test_dataset * W_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))};
        nbMaxComp = size(W_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}, 2);
        nbCarac = [nbMaxComp ceil(nbMaxComp/2) 3];
        for idxImagProj = 1:length(nbCarac)
        
            imagRecomp = test_dataset_proj(:,1:nbCarac(idxImagProj)) * W_pca{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}(:,1:1:nbCarac(idxImagProj))';
            reshapedimagRecomp = reshape(uint8(double(imagRecomp(idxImagOrig,:)))',112,92);
            subplot(length(nbSubjects) * length(nbImages),length(nbImages) + 2,(((idxNbPersons-1)*2+idxImage)-1)*4+1+idxImagProj);
            imshow(reshapedimagRecomp);
            title(sprintf('pca,%d,%d,%d',[nbSubjects(idxNbPersons), nbImages(idxImage),nbCarac(idxImagProj)]));
        end
    end
end

figure('NumberTitle', 'off', 'Name', 'Examples of face reconstruction using LDA');
for idxNbPersons = 1:length(nbSubjects)
    for idxImage = 1:length(nbImages)
        idxImagOrig = randi(size(test_dataset,1),1); 
        origImag = test_dataset(idxImagOrig,:);
        reshapedImagOrig = reshape(uint8(double(test_dataset(idxImagOrig,:)))',112,92);
        subplot(length(nbSubjects) * length(nbImages),length(nbImages) + 2,(((idxNbPersons-1)*2+idxImage)-1)*4+1);
        imshow(reshapedImagOrig);
        title(sprintf('lda,%d,%d,orig',[nbSubjects(idxNbPersons), nbImages(idxImage)]));
        test_dataset_proj = test_dataset * W_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))};
        nbMaxComp = size(W_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}, 2);
        nbCarac = [nbMaxComp ceil(nbMaxComp/2) 3];
        for idxImagProj = 1:length(nbCarac)
            imagRecomp = test_dataset_proj(:,1:nbCarac(idxImagProj)) * W_lda{(nbSubjects(idxNbPersons)),(nbImages(idxImage))}(:,1:1:nbCarac(idxImagProj))';
            reshapedimagRecomp = reshape(uint8(double(imagRecomp(idxImagOrig,:)))',112,92);
            subplot(length(nbSubjects) * length(nbImages),length(nbImages) + 2,(((idxNbPersons-1)*2+idxImage)-1)*4+1+idxImagProj);
            imshow(reshapedimagRecomp);
            title(sprintf('lda,%d,%d,%d',[nbSubjects(idxNbPersons), nbImages(idxImage),nbCarac(idxImagProj)]));
        end
    end
end

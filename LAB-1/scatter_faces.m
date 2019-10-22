% Mohammadamin Abbasnejad
% Lab 2
addpath(genpath('prtools\'));
clear all;
close all;
clc;
prwarning(0);


nbPersons = [5 20 40]; 
nbImages = [2 5]; 

load('test_dataset.mat');
load('train_dataset.mat','W*');

%%
% for PCA

for idxNbPersons = 1:length(nbPersons)
    for idxImage = 1:length(nbImages)
        figure('NumberTitle', 'off', 'Name', sprintf('Analyse de la dispersion, original vs PCA, %d personnes, %d images', [nbPersons(idxNbPersons), nbImages(idxImage)]));
        subplot(2,2,1);
        scatterd(test_dataset);
        title(sprintf('%d,%d,orig',[nbPersons(idxNbPersons), nbImages(idxImage)]));
        test_dataset_proj = test_dataset * W_pca{(nbPersons(idxNbPersons)),(nbImages(idxImage))};
        nbMaxComp = size(W_pca{(nbPersons(idxNbPersons)),(nbImages(idxImage))}, 2);
        nbCarac = [nbMaxComp ceil(nbMaxComp/2) 3]; 
        for idxImagProj = 1:length(nbCarac)
            
            imagRecomp = test_dataset_proj(:,1:nbCarac(idxImagProj)) * W_pca{(nbPersons(idxNbPersons)),(nbImages(idxImage))}(:,1:1:nbCarac(idxImagProj))';
            subplot(2,2,1+idxImagProj);
            scatterd(imagRecomp);
            title(sprintf('pca,%d,%d,%d',[nbPersons(idxNbPersons), nbImages(idxImage),nbCarac(idxImagProj)]));
        end
    end
end

for idxNbPersons = 1:length(nbPersons)
    for idxImage = 1:length(nbImages)
        figure('NumberTitle', 'off', 'Name', sprintf('Analyse de la dispersion, original vs LDA, %d personnes, %d images', [nbPersons(idxNbPersons), nbImages(idxImage)]));
        subplot(2,2,1);
        scatterd(test_dataset);
        title(sprintf('%d,%d,orig',[nbPersons(idxNbPersons), nbImages(idxImage)]));
        test_dataset_proj = test_dataset * W_lda{(nbPersons(idxNbPersons)),(nbImages(idxImage))};
        nbMaxComp = size(W_lda{(nbPersons(idxNbPersons)),(nbImages(idxImage))}, 2);
        nbCarac = [nbMaxComp ceil(nbMaxComp/2) 3]; 
        for idxImagProj = 1:length(nbCarac)

            imagRecomp = test_dataset_proj(:,1:nbCarac(idxImagProj)) * W_pca{(nbPersons(idxNbPersons)),(nbImages(idxImage))}(:,1:1:nbCarac(idxImagProj))';
            subplot(2,2,1+idxImagProj);
            scatterd(imagRecomp);
            title(sprintf('lda,%d,%d,%d',[nbPersons(idxNbPersons), nbImages(idxImage),nbCarac(idxImagProj)]));
        end
    end
end
% Mohammadamin Abbasnejad
% Lab 2
% -------------------------------------------------------------------------
% Extracting image data from faces from pgm files
% And save in a prdataset format
% -------------------------------------------------------------------------
addpath(genpath('prtools\'));
clear all;
close all;
clc;

%% path to images
path_to_images = 'C:\Users\amin\Documents\MATLAB\Database\att_faces\';
image_extension = '*.pgm';
dirperson = dir(strcat(path_to_images, 's*'));
nperson = length(dirperson);    


for id_subj = 1:nperson
    labels = [];
    images = [];
   
    dirimage_all = dir(strcat(path_to_images, '/s', num2str(id_subj)));
    dirimage_all=dirimage_all(~ismember({dirimage_all.name},{'.','..'}));
    nimage = length(dirimage_all);
    for idx_image = 1:nimage
       
       currentfilename = strcat(dirimage_all(idx_image).folder, '/', dirimage_all(idx_image).name);
       currentimage = imread(currentfilename);
       currentimage = currentimage(:)';
       delete(currentfilename); 
   end
    images = prdataset(double(images), labels);
    save(strcat(path_to_images, '/s', num2str(id_subj), '/s', num2str(id_subj)), 'images')
    strcat(path_to_images, '/s', num2str(id_subj), '/s', num2str(id_subj))
end
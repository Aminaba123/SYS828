% Mohammadamin Abbasnejad
% Lab 3
% -------------------------------------------------------------------------
% Extraction of images of faces saved in prdataset format
% -------------------------------------------------------------------------

function dataset = read_faces(path, nperson, nphotos)
% Input :
% Output :
% dataset is a prdataset structure that contains hotos of p erson


dataset = prdataset([]);
for idx_person = nperson
    load(strcat(path, 's', num2str(idx_person), '/s', num2str(idx_person), '.mat'));
    
    images = images(nphotos, :);
    
    dataset = [dataset; images];
end

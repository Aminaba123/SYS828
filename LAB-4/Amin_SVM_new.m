clc
clear all

disp ('Loading data ...')
load('Grille_features_95')


TrainAll = Train_features;
TrainAll_label = train_labels;

TestAll = Test_features;
TestAll_label = test_labels;

disp('Data Preparating: Train set, Validation set and Test set')

class_size = 600;
vald_class_size = 600*1/3;
train_class_size = 600*2/3;

Train_set_label = [];
Train_set = [];
Vald_set_label = [];
Vald_set = [];
for i = 1 :10
    Train_set_label = [Train_set_label, TrainAll_label(((i-1)*600)+1:((i)*600)-vald_class_size)];
    Train_set = [Train_set; TrainAll(((i-1)*600)+1:((i)*600)-vald_class_size, :)];
    
    Vald_set_label = [Vald_set_label, TrainAll_label(((i-1)*600)+1+train_class_size:((i)*600))];
    Vald_set = [Vald_set; TrainAll(((i-1)*600)+1+train_class_size:((i)*600), :)];
end
disp('=================================================')

%% One vs All
class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9];

Err_all = zeros(5,7);
% C_list = [1 10  100 500 1000 2000 5000 7000 8500 10000];
% sigma_list = [10^-8 10^-7 10^-6 10^-5 10^-4 .001 .005 .01 .1 1];
C_sigma_mat = {};
cc = 1;
for C = [1 10 100 1000 10000]
    sigma_sigma = 1;
    for sigma = [10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 10^0]
        score_class = zeros(10, 1);
        for i = 1: 10
            k = class_list(i);
            idx_class_val = find(Vald_set_label == k);
            Vald_label_bin = zeros(size(Vald_set_label))-1;
            Vald_label_bin(idx_class_val) = 1;

            idx_class_train = find(Train_set_label == k);
            Train_set_label_bin = zeros(size(Train_set_label ))-1;
            Train_set_label_bin(idx_class_train) = 1;

            idx_class_test = find(TestAll_label == k);
            TestAll_label_bin = zeros(size(TestAll_label))-1;
            TestAll_label_bin (idx_class_test) = 1;
            
            % fprintf(' Trainin SVM for the class:%d\n', i)
            SVMStruct = svmtrain(Train_set, Train_set_label_bin, 'kernel_function', 'rbf', 'rbf_sigma', 10^(((-sigma)+1)/30), 'showplot',false, 'autoscale', false, 'boxconstraint', (10^(C-2))/50);
            % fprintf(' Testing SVM for the class:%d\n', i)
            pred = svmclassify(SVMStruct, Vald_set);
            %score = 1 - (length(find(pred == Vald_label_bin')/length(Vald_label_bin')));
            score = 1 - (length(find(pred == Vald_label_bin'))/length(Vald_label_bin'));
            score_class(i,1) = score;
        end
        

        fprintf('Err for C:%d and sigma:%d is:%g\n',C, sigma, mean(score_class))
        Err_all(cc, sigma_sigma) = mean(score_class);
        C_sigma_mat{cc, sigma_sigma} = {C, sigma};
        sigma_sigma = sigma_sigma + 1;
    end
    cc = cc + 1;
end



%% Visualizing Err with respect to C and sigma

C = 1:5; %C_list;
sigma = 1:7; %sigma_list;
surf( Err_all)

xlabel('C')
ylabel('sigma')
zlabel('Error ')
title('Error as function of c and sigma')
%% Training with the best sigma and C values
score_class_opt = zeros(10, 1);

[m n] = min(Err_all(:));
[C_ind sigma_ind] = ind2sub(size(Err_all), n);
C_opt_sigma_opt = C_sigma_mat{C_ind, sigma_ind};

C_opt = C_opt_sigma_opt{1};
sigma_opt = C_opt_sigma_opt{1, 2};

class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

disp('Training and Testing with the opt C and sigma')

for i = 1: 10
    
    k = class_list(i);
    idx_class_train = find(TrainAll_label == k);
    TrainAll_label_bin = zeros(size(TrainAll_label ))-1;
    TrainAll_label_bin(idx_class_train) = 1;

    idx_class_test = find(TestAll_label == k);
    TestAll_label_bin = zeros(size(TestAll_label))-1;
    TestAll_label_bin (idx_class_test) = 1;

    options.MaxIter = 30000;
    % fprintf(' Trainin SVM for the class:%d\n', i)
    tic
    SVMStruct = svmtrain(TrainAll, TrainAll_label_bin, 'Options', options,  'kernel_function', 'rbf', 'rbf_sigma', sigma_opt, 'showplot',false, 'autoscale', false, 'boxconstraint', C_opt);
    training_time = toc;
    tic
    % fprintf(' Testing SVM for the class:%d\n', i)
    pred = svmclassify(SVMStruct, TestAll);
    testing_time = toc;
    fprintf('Training time is : %d\n', training_time)
    fprintf('Testing time is : %d\n', testing_time)
    score = 1 - (length(find(pred == TestAll_label_bin'))/length(TestAll_label_bin'));
    score_class_opt(i,1) = score;
end
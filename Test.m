data_dir = 'UIUC_Sports';
img_dir = [data_dir '/event_img'];

fea_dir = [data_dir '/feature'];         mkdir(fea_dir);
fea_flip_dir = [data_dir '/feature_flip']; mkdir(fea_flip_dir);

svm.c = 1;
nRounds = 10;       % experiment times
tr_num  = 100;       % number of training image of each class

useGpu = true;
useFlip = true;
  
addpath('D:\Toolbox\matconvnet-master\matlab');  % the path of MatConvNet toolbox
addpath('D:\Toolbox\liblinear-1.94\matlab\');    % the path of LibLinear toolbox
vl_setupnn();

% addpath('/home/local/USHERBROOKE/luoz3301/toolbox/matconvnet-1.0-beta8/matlab');
% addpath('/home/local/USHERBROOKE/luoz3301/toolbox/liblinear-1.96/linux');


net = load('D:\Toolbox\matconvnet-1.0-beta7\matlab\Model\imagenet-caffe-ref.mat');
if useGpu
    net = vl_simplenn_move(net,'gpu');
end

% compute features
if ~exist([data_dir '/database.mat'])
    database = retr_database_dir(img_dir);

    database.img_dir = img_dir;
    database.fea_dir = fea_dir;
    database.fea_flip_dir = fea_flip_dir;

    database = computeFeature(database, net, useGpu, useFlip);
    save([data_dir '/database.mat'],'database');
else
    load([data_dir '/database.mat']);
end

% classification
fprintf('\n Testing...\n');
accuracy = svm_classify(database, nRounds, tr_num, svm.c, useFlip);


Ravg = mean(accuracy);                  % average recognition rate
Rstd = std(accuracy);                   % standard deviation of the recognition rate

fprintf('===============================================\n');
fprintf('Average classification accuracy: %.2f%%\n', Ravg*100);
fprintf('Standard deviation: %.2f\n', Rstd*100);
fprintf('===============================================\n');

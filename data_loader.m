% Sai Satya Charan Malladi
% AEROSP 567 Fall 21
% Final Project

% data_loader.m
% file to pre-process data

%% Begin 

% load the walking data of subject 02 
load('02_02_moc.mat')
data_dense = wsMoc.Dof;

% sample every 4 frames to generate sparse data
sample_rate = 4;
data_sparse = data_dense(:,1:sample_rate:length(data_dense));

% subtract mean from the sparse data
data_sparse = data_sparse - mean(data_sparse,2);
save('data_walking_sparse.mat','data_sparse')


% load the running data of subject 02 
load('02_03_moc.mat')
data_dense = wsMoc.Dof;

% sample every 4 frames to generate sparse data
sample_rate = 4;
data_sparse = data_dense(:,1:sample_rate:length(data_dense));

% subtract mean from the sparse data
data_sparse = data_sparse - mean(data_sparse,2);
save('data_running_sparse.mat','data_sparse')


% load the jumping data of subject 02 
load('02_04_moc.mat')
data_dense = wsMoc.Dof;

% sample every 4 frames to generate sparse data
sample_rate = 4;
data_sparse = data_dense(:,1:sample_rate:length(data_dense));

% subtract mean from the sparse data
data_sparse = data_sparse - mean(data_sparse,2);
save('data_jumping_sparse.mat','data_sparse')
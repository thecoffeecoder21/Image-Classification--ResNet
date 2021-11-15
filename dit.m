clc
clear all
close all
fpath='C:\Users\amank\Desktop\Image Classification -ResNet';
data = fullfile(fpath,'PreProcessedImages');
tdata = imageDatastore(data,'includesubfolders',true,'LabelSource','foldername')
count = tdata.countEachLabel
%split data
[trainingdata testdata]= splitEachLabel(tdata,0.8,'randomized')

% Load pre-Trained Network
net=resnet50
%net=alexnet
layers=[imageInputLayer([227 227])
    net(2:end-3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer()
    ];
opt=trainingOptions('sgdm','Maxepoch',50,'InitialLearnRate',0.001,'Plots','training-progress');
%opt=trainingOptions('sgdm','Maxepoch',20,'InitialLearnRate',0.001);
training=trainNetwork(trainingdata,layers,opt);

%save training

%analyse

%analyzeNetwork(training)
%load training.mat

allclass=[];
for i=1:length(testdata.Labels)
    I=readimage(testdata,i);
    class=classify(training,I);
    allclass=[allclass class];
    figure(1),
    subplot(25, 10,i)
    imshow(I)
    title(char(class))
end


    
    
    
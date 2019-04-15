clear all
close all
clc

net = resnet50();
setDir = 'Sorted_Cars_By_Type_15cm_24px-exc_v5-marg-32_expanded/Potsdam/';
imgSets = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
[trainingSet, testSet] = splitEachLabel(imgSets, 0.3, 'randomize');

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ... 
    'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', ...
    'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');



%https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html

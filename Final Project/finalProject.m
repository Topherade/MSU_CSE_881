clear all
close all
clc

net = resnet50();
setDir = 'Sorted_Cars_By_Type_15cm_24px-exc_v5-marg-32_expanded/Potsdam/';
imgSets = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
[train, test] = splitEachLabel(imgSets, 0.5, 'randomize');

imageSize = net.Layers(1).InputSize;
aug_train = augmentedImageDatastore(imageSize, train, ...
    'ColorPreprocessing', 'gray2rgb');
aug_test = augmentedImageDatastore(imageSize, test, ... 
    'ColorPreprocessing', 'gray2rgb');

trainFeatures = activations(net, aug_train, 'fc1000', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = train.Labels;

% Train multiclass SVM classifier using a fast linear solver
classifier = fitcecoc(trainFeatures, trainingLabels, 'Learners', ...
    'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Extract test features using the CNN
testFeatures = activations(net, aug_test, 'fc1000', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = test.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictLabels)

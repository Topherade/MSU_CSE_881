clear all
close all
clc

net = resnet50();

figure
plot(net)
title('ResNet-50');
set(gca,'YLim',[150 170]);

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
testFeatures = activations(net, aug_test, 'fc1000', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels
trainingLabels = train.Labels;
% Get test labels
testLabels = test.Labels;

% Train multiclass SVM classifier using a fast linear solver
t = templateSVM('Standardize',true,'SaveSupportVectors',true);
mdl = fitcecoc(trainFeatures, trainingLabels, 'Learners', ...
    t, 'Coding', 'onevsall', 'ObservationsIn', 'columns');

length = size(mdl.CodingMatrix,2); % Number of SVMs
sv = cell(length,1); % Preallocate for support vector indices
for j = 1:length
    SVM = mdl.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end

figure
gscatter(trainFeatures(1,:),trainFeatures(2,:),trainingLabels);
hold on
markers = {'ko','ro','bo'};
for j = 1:length
    svs = sv{j};
    plot(svs(:,1),svs(:,2),markers{j},...
        'MarkerSize',10 + (j - 1)*3);
end
legend({'Car', 'No Car', 'SVM 1'},'Location','Best')

% Pass CNN image features to trained classifier
predictLabels = predict(mdl, testFeatures, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictLabels)

clear all
close all
clc

setDir = 'Sorted_Cars_By_Type_15cm_24px-exc_v5-marg-32_expanded\Potsdam\';
imgSets = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

[trainingSet,testSet] = splitEachLabel(imgSets,0.3,'randomize');

bag = bagOfFeatures(trainingSet);

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confMatrix = evaluate(categoryClassifier,testSet)
mean(diag(confMatrix))

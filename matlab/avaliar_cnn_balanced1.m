load('cnn_smoke_classifier_balanced.mat');

cnnPath = 'C:\Users\manue\Desktop\Projeto LP\dataset_cnn_balanced';
imds = imageDatastore(cnnPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[~, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
imdsVal.ReadFcn = @(x) imresize(imread(x), [224 224]);

predictedLabels = classify(net, imdsVal);
trueLabels = imdsVal.Labels;

figure;
confusionchart(trueLabels, predictedLabels);

accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Accuracy: ', num2str(accuracy*100), '%']);
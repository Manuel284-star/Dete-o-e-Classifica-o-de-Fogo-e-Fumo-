load('cnn_smoke_classifier.mat');

% Carregar dados de validação para testar
cnnPath = 'C:\Users\manue\Desktop\Projeto LP\dataset_cnn';
imds = imageDatastore(cnnPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[~, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
imdsVal.ReadFcn = @(x) imresize(imread(x), [224 224]);

% Prever
predictedLabels = classify(net, imdsVal);
trueLabels = imdsVal.Labels;

% Matriz de confusão
figure;
confusionchart(trueLabels, predictedLabels);

% Calcular métricas
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Accuracy: ', num2str(accuracy*100), '%']);

% Ver quantas previu de cada classe
disp('Previsões por classe:');
countEachLabel(predictedLabels)
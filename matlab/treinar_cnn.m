% Carregar dataset de imagens
cnnPath = 'C:\Users\manue\Desktop\Projeto LP\dataset_cnn';
imds = imageDatastore(cnnPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Ver distribuição de classes
disp('Distribuição de classes:');
countEachLabel(imds)

% Dividir em treino (80%) e validação (20%)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% Redimensionar imagens para 224x224 (tamanho de input da rede)
imdsTrain.ReadFcn = @(x) imresize(imread(x), [224 224]);
imdsVal.ReadFcn = @(x) imresize(imread(x), [224 224]);

% Definir arquitetura da rede CNN
layers = [
    imageInputLayer([224 224 3], 'Name', 'input', 'Normalization', 'zerocenter')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
    ];

% Opções de treino
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16, ...
    'ValidationData', imdsVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch');

% Treinar
disp('A iniciar treino da CNN...');
[net, info] = trainNetwork(imdsTrain, layers, options);

% Guardar
save('cnn_smoke_classifier.mat', 'net', 'info');
disp('Treino concluído e modelo guardado!');
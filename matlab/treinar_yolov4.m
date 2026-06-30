% Carregar dataset preparado
load('trainingData.mat');
disp(['Imagens de treino: ', num2str(height(trainingData))]);

% Usar subconjunto pequeno para teste
trainingData = trainingData(1:2000, :);
disp(['A usar subconjunto de: ', num2str(height(trainingData)), ' imagens']);

% Definir classes e anchor boxes
classNames = {'smoke'};
anchorBoxes = {[12 16; 19 36; 40 28], ...
               [36 75; 76 55; 72 146], ...
               [142 110; 192 243; 459 401]};

% Carregar YOLOv4 pré-treinado
detector = yolov4ObjectDetector('csp-darknet53-coco', classNames, anchorBoxes);
disp('Modelo carregado com sucesso!');

% Converter para datastore
imds = imageDatastore(trainingData.imageFilename);
blds = boxLabelDatastore(trainingData(:, 2:end));
ds = combine(imds, blds);
disp('Datastore criado!');

% Opções de treino reduzidas
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 2, ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'Shuffle', 'every-epoch');

% Treinar
disp('A iniciar treino...');
[trainedDetector, info] = trainYOLOv4ObjectDetector(ds, detector, options);

save('yolov4_smoke.mat', 'trainedDetector', 'info');
disp('Treino concluído!');
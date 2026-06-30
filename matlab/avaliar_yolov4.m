% Carregar modelo treinado
load('yolov4_smoke.mat');

% Preparar dados de validação
datasetPath = 'C:\Users\manue\Desktop\Projeto LP\dataset';
imgValPath = fullfile(datasetPath, 'images', 'val');
lblValPath = fullfile(datasetPath, 'labels', 'val');

imgFiles = dir(fullfile(imgValPath, '*.jpg'));
disp(['Imagens de validação: ', num2str(length(imgFiles))]);

% Converter labels de validação
imageFilenames = {};
bboxes = {};

for i = 1:length(imgFiles)
    imgName = imgFiles(i).name;
    imgPath = fullfile(imgValPath, imgName);
    info = imfinfo(imgPath);
    W = info.Width;
    H = info.Height;

    lblName = strrep(imgName, '.jpg', '.txt');
    lblPath = fullfile(lblValPath, lblName);

    boxes = [];
    if isfile(lblPath)
        fid = fopen(lblPath, 'r');
        line = fgetl(fid);
        while ischar(line)
            vals = str2num(line);
            if length(vals) == 5
                cx = vals(2) * W;
                cy = vals(3) * H;
                bw = vals(4) * W;
                bh = vals(5) * H;
                x = cx - bw/2;
                y = cy - bh/2;
                boxes = [boxes; x y bw bh];
            end
            line = fgetl(fid);
        end
        fclose(fid);
    end

    if ~isempty(boxes)
        imageFilenames{end+1} = imgPath;
        bboxes{end+1} = boxes;
    end

    if mod(i, 500) == 0
        disp(['Processadas: ', num2str(i)]);
    end
end

valData = table(imageFilenames', bboxes', 'VariableNames', {'imageFilename', 'smoke'});
disp(['Imagens de val com labels: ', num2str(height(valData))]);

% Usar só 100 imagens para avaliação
valDataSmall = valData(1:100, :);

% Criar datastore de imagens
imdsVal = imageDatastore(valDataSmall.imageFilename);

% Criar datastore de labels
blds = boxLabelDatastore(valDataSmall(:, 2:end));

% Combinar
dsVal = combine(imdsVal, blds);

% Correr deteção
disp('A correr deteção...');
detectionResults = detect(trainedDetector, imdsVal, 'MiniBatchSize', 1);

% Avaliar
results = evaluateObjectDetection(detectionResults, dsVal);
disp(results);
disp(results.ClassMetrics);
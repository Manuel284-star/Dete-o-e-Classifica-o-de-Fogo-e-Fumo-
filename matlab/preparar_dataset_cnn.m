% Preparar dataset para classificação CNN
datasetPath = 'C:\Users\manue\Desktop\Projeto LP\dataset';
imgTrainPath = fullfile(datasetPath, 'images', 'train');
lblTrainPath = fullfile(datasetPath, 'labels', 'train');

% Criar pastas de destino
cnnPath = 'C:\Users\manue\Desktop\Projeto LP\dataset_cnn';
smokePath = fullfile(cnnPath, 'smoke');
noSmokePath = fullfile(cnnPath, 'no_smoke');
mkdir(smokePath);
mkdir(noSmokePath);

imgFiles = dir(fullfile(imgTrainPath, '*.jpg'));
disp(['Total de imagens: ', num2str(length(imgFiles))]);

% Usar só um subconjunto para ser viável sem GPU
numToUse = 4000;
idx = randperm(length(imgFiles), min(numToUse, length(imgFiles)));

countSmoke = 0;
countNoSmoke = 0;

for k = 1:length(idx)
    i = idx(k);
    imgName = imgFiles(i).name;
    imgPath = fullfile(imgTrainPath, imgName);

    lblName = strrep(imgName, '.jpg', '.txt');
    lblPath = fullfile(lblTrainPath, lblName);

    hasSmoke = false;
    if isfile(lblPath)
        fid = fopen(lblPath, 'r');
        line = fgetl(fid);
        if ischar(line) && ~isempty(line)
            hasSmoke = true;
        end
        fclose(fid);
    end

    if hasSmoke
        copyfile(imgPath, fullfile(smokePath, imgName));
        countSmoke = countSmoke + 1;
    else
        copyfile(imgPath, fullfile(noSmokePath, imgName));
        countNoSmoke = countNoSmoke + 1;
    end

    if mod(k, 500) == 0
        disp(['Processadas: ', num2str(k), '/', num2str(length(idx))]);
    end
end

disp(['Imagens com fumo: ', num2str(countSmoke)]);
disp(['Imagens sem fumo: ', num2str(countNoSmoke)]);
disp('Dataset CNN pronto!');
% Preparar dataset CNN balanceado
datasetPath = 'C:\Users\manue\Desktop\Projeto LP\dataset';
imgTrainPath = fullfile(datasetPath, 'images', 'train');
lblTrainPath = fullfile(datasetPath, 'labels', 'train');

cnnPath = 'C:\Users\manue\Desktop\Projeto LP\dataset_cnn_balanced';
smokePath = fullfile(cnnPath, 'smoke');
noSmokePath = fullfile(cnnPath, 'no_smoke');
mkdir(smokePath);
mkdir(noSmokePath);

imgFiles = dir(fullfile(imgTrainPath, '*.jpg'));
disp(['Total de imagens: ', num2str(length(imgFiles))]);

% Separar todas as imagens em smoke/no_smoke
smokeFiles = {};
noSmokeFiles = {};

for i = 1:length(imgFiles)
    imgName = imgFiles(i).name;
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
        smokeFiles{end+1} = imgName;
    else
        noSmokeFiles{end+1} = imgName;
    end

    if mod(i, 5000) == 0
        disp(['Classificadas: ', num2str(i), '/', num2str(length(imgFiles))]);
    end
end

disp(['Total com fumo: ', num2str(length(smokeFiles))]);
disp(['Total sem fumo: ', num2str(length(noSmokeFiles))]);

% Balancear: mesmo número de cada classe
numPerClass = min(length(smokeFiles), length(noSmokeFiles));
numPerClass = min(numPerClass, 2000);

smokeIdx = randperm(length(smokeFiles), numPerClass);
noSmokeIdx = randperm(length(noSmokeFiles), numPerClass);

disp(['A copiar ', num2str(numPerClass), ' imagens por classe...']);

for k = 1:numPerClass
    copyfile(fullfile(imgTrainPath, smokeFiles{smokeIdx(k)}), fullfile(smokePath, smokeFiles{smokeIdx(k)}));
    copyfile(fullfile(imgTrainPath, noSmokeFiles{noSmokeIdx(k)}), fullfile(noSmokePath, noSmokeFiles{noSmokeIdx(k)}));

    if mod(k, 500) == 0
        disp(['Copiadas: ', num2str(k), '/', num2str(numPerClass)]);
    end
end

disp('Dataset balanceado pronto!');
disp(['Smoke: ', num2str(numPerClass), ' | No smoke: ', num2str(numPerClass)]);
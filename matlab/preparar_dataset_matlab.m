% Preparar dataset para MATLAB YOLOv4
datasetPath = 'C:\Users\manue\Desktop\Projeto LP\dataset';
imgTrainPath = fullfile(datasetPath, 'images', 'train');
lblTrainPath = fullfile(datasetPath, 'labels', 'train');

% Listar imagens
imgFiles = dir(fullfile(imgTrainPath, '*.jpg'));
disp(['Total de imagens: ', num2str(length(imgFiles))]);

% Converter labels YOLO para formato MATLAB
imageFilenames = {};
bboxes = {};

for i = 1:length(imgFiles)
    imgName = imgFiles(i).name;
    imgPath = fullfile(imgTrainPath, imgName);
    
    % Ler tamanho da imagem
    info = imfinfo(imgPath);
    W = info.Width;
    H = info.Height;
    
    % Ler label correspondente
    lblName = strrep(imgName, '.jpg', '.txt');
    lblPath = fullfile(lblTrainPath, lblName);
    
    boxes = [];
    if isfile(lblPath)
        fid = fopen(lblPath, 'r');
        line = fgetl(fid);
        while ischar(line)
            vals = str2num(line);
            if length(vals) == 5
                % Converter de YOLO (cx cy w h) para MATLAB (x y w h)
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
    
    if mod(i, 1000) == 0
        disp(['Processadas: ', num2str(i), '/', num2str(length(imgFiles))]);
    end
end

% Criar tabela de treino
trainingData = table(imageFilenames', bboxes', 'VariableNames', {'imageFilename', 'smoke'});
save('trainingData.mat', 'trainingData');
disp(['Dataset pronto! Total com labels: ', num2str(height(trainingData))]);
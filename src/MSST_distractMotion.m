% function [] = MSST_distractMotion(times_number, type1, type2)
close all
%%  Configuration
participant = 'P_GQY_side'; 
type = 'eatFetch_mute';
times_number = 8;
duration = 5;
isSaveSTFTFigure = 0;
isSave = 0;
showSpectrogram = 1;
showExtractFigure = 1;
%%  changeable path
path_voice2 = strcat(participant,'\CollectedData_eatFetch\CollectedData_',type,'\CollectedData\WAV\SingleWav');
%%  verify
path_voice3 = '.wav';
path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
if (isfile(path_voice)) == 0
    return 
end
%%  Read SingleWav
[x, Fs] = audioread(path_voice); 
x = x(3*Fs:end-3*Fs);  %x = x(5*Fs:end-2*Fs);  
LenData = length(x); % the length of wav
time = (0 : LenData-1) / Fs; 
shape = size(x);
fftRealArray = zeros(shape)'; 
LenBatch = 3584; % android microphone buffer size is 3584
Amp = zeros(floor(LenData/LenBatch), 4096); % amp contains all the bins 2048
%%  STFT
[S,F,T,P] = spectrogram(x, hanning(round(fix(23.1*0.01*Fs)/2)), round(fix(23.1*0.01*Fs)/2.5), fix(23.1*0.01*Fs*2.5), Fs);

% fetch 19850-20150 Hz in spectrogram
F_step = F(3) - F(2);
F_first_index = floor(19900/F_step);
F_end_index = floor(20100/F_step);
P = P(F_first_index:F_end_index,:);
output = 10*log10(abs(P));

% norm
% output_mean = mean(mean(output));
% [output_f, output_t] = size(output);
% for i = 1:output_f
%     for j = 1:output_t
%         if (output(i,j) < output_mean)
%             output(i,j) = output_mean;
%         end
%     end
% end

% show spectrogram
if (showSpectrogram)
    figure();
    imagesc(output);
    % imagesc(T,F(F_first_index:F_end_index),output); 
    xlabel('Time (Seconds)'); ylabel('Hz');
end

% sum frequency
% Min-max normalize
output_columnSum = zscore(sum(output)); %列求和
output_columnSum = smooth(output_columnSum(:), 10);
% output_columnSum = output_columnSum + abs(min(output_columnSum));
% output_columnSum(1:fix(length(output_columnSum)/2)) = output_columnSum(1:fix(length(output_columnSum)/2))*0.3;
% save points > threshold
cloumnSum_sort = sort(output_columnSum);
cloumnSum_index = fix(length(cloumnSum_sort)*0.8);
threshold = cloumnSum_sort(cloumnSum_index);
peak_index = [];
for i = 1:length(output_columnSum)
    if (output_columnSum(i) > threshold)
        peak_index = [peak_index i];
    end
end
figure();
plot(T, output_columnSum);
xlabel('Time (s)');
ylabel('Motion Y');
hold on;
plot(T(peak_index), output_columnSum(peak_index), 'r*', 'LineWidth', 2, 'MarkerSize', 10);
% figureSavePath1 = strcat(participant,'\Figure\distractMotion\');
% figureSavePath2  = 'cloumnSum_motion.jpg';
% figureSavePath_1 = sprintf('%s%d%s',figureSavePath1, times_number, figureSavePath2);
% saveas(gcf, figureSavePath_1);

% Extract 2 motion  *** Finesse turnBack & pickUp ***
% max_T = max(T(peak_index));
% min_T = min(T(peak_index));
% half_time = (max(T(peak_index)) - min(T(peak_index)))/2 + min_T;
% motion_center = [0 0];
% tmp_1 = [];
% tmp_2 = [];
% for i = 1:length(peak_index)
%     if (T(peak_index(i)) < half_time)
%         tmp_1 = [tmp_1 T(peak_index(i))];
%     else
%         tmp_2 = [tmp_2 T(peak_index(i))];
%     end
% end
% motion_center(1) = mean(tmp_1);
% motion_center(2) = mean(tmp_2);


% Extract 2 motion
motion_interval = 3; %3s
tmp = [];
motion_center = [];
for i = 1:length(peak_index)-1    
    if (T(peak_index(i+1)) - T(peak_index(i)) > motion_interval)
        if (i == length(peak_index)-1)
            tmp = [tmp T(peak_index(i))];
            motion_center = [motion_center mean(tmp)];
            motion_center = [motion_center T(peak_index(i+1))];
        else 
            tmp = [tmp T(peak_index(i))];
            motion_center = [motion_center mean(tmp)];
            tmp = [];
        end

    else
        if (i == length(peak_index)-1)
            tmp = [tmp T(peak_index(i))];
            tmp = [tmp T(peak_index(i+1))];
            motion_center = [motion_center mean(tmp)];
        else 
            tmp = [tmp T(peak_index(i))];
        end
    end
end


    
% set edge 
if (motion_center(1) - duration/2 < 0)
    motion_1_front = 0.2;
    motion_1_rear = duration;
else 
    motion_1_front = roundn(motion_center(1) - duration/2, -3);
    motion_1_rear = roundn(motion_center(1) + duration/2, -3);
end

if (motion_center(2) + duration/2 > max(time))
    motion_2_front = max(time) - duration - 0.2;
    motion_2_rear = max(time) - 0.2;
else 
    motion_2_front = roundn(motion_center(2) - duration/2, -3);
    motion_2_rear = roundn(motion_center(2) + duration/2, -3);
end


% show extract figure
if (showExtractFigure)
    figure();
    plot(T, output_columnSum);
    xlabel('Time (s)');
    ylabel('Motion Y');
    hold on;
    plot(T(peak_index), output_columnSum(peak_index), 'r*', 'LineWidth', 2, 'MarkerSize', 10);
    rectangle('Position',[motion_1_front,min(output_columnSum),motion_1_rear-motion_1_front,max(output_columnSum)-min(output_columnSum)],'LineWidth',1,'EdgeColor','r' )
    rectangle('Position',[motion_2_front,min(output_columnSum),motion_2_rear-motion_2_front,max(output_columnSum)-min(output_columnSum)],'LineWidth',1,'EdgeColor','r' )
end

motion_1_front_index = fix(motion_1_front/(T(2)-T(1)));
motion_1_rear_index = fix(motion_1_rear/(T(2)-T(1)));
motion_2_front_index = fix(motion_2_front/(T(2)-T(1)));
motion_2_rear_index = fix(motion_2_rear/(T(2)-T(1)));
motion_1 = output(:,motion_1_front_index:motion_1_rear_index);
motion_2 = output(:,motion_2_front_index:motion_2_rear_index);

% save motion figure
if (isSaveSTFTFigure)
    imagesc(motion_1)
    figureSavePath1 = strcat(participant,'\Figure\distractMotion\');
    figureSavePath2  = '2_motion1.jpg';
    figureSavePath_1 = sprintf('%s%d%s',figureSavePath1, times_number, figureSavePath2);
    saveas(gcf, figureSavePath_1);
    imagesc(motion_2)
    figureSavePath2  = '2_motion2.jpg';
    figureSavePath_2 = sprintf('%s%d%s',figureSavePath1, times_number, figureSavePath2);
    saveas(gcf, figureSavePath_2);
end

%% MSSTFeature
if (isSave)
    % motion_1
    [mappedD1, ~] = compute_mapping(squeeze(motion_1(1:89,:)'), 'PCA', 1);%上半段
    [mappedD2, ~] = compute_mapping(squeeze(motion_1(89:end,:)'), 'PCA', 1);%下半段
    mappedD1 = smooth(mappedD1,10);
    mappedD2 = smooth(mappedD2,10);
    mappedD1 = zscore(mappedD1);
    mappedD2 = zscore(mappedD2);
    if (max(mappedD1)<2)
    mappedD1 = -mappedD1;
    end
    if (max(mappedD2)<2)
    mappedD2 = -mappedD2;
    end
    MSST_motion1 = real(IMSST_W(mappedD1,40,10));
    MSST_motion2 = real(IMSST_W(mappedD2,40,10));
    MSST_motion1 = MSST_motion1(1:60,:);
    MSST_motion2 = MSST_motion2(1:60,:);
    % % save MSST figure
    % figure();
    % subplot(2,1,1);
    % imagesc(MSST_motion1);
    % subplot(2,1,2);
    % imagesc(MSST_motion2);
    % figureSavePath1 = strcat(participant,'\Figure\4motion_MSST\');
    % figureSavePath2  = 'motion1.jpg';
    % figureSavePath_1 = sprintf('%s%d%s',figureSavePath1, times_number, figureSavePath2);
    % saveas(gcf, figureSavePath_1);
    % close all
    voicePath_store0  = strcat(participant,'\MSSTFeature\distractMotion\',type1,'\MSSTFeature');
    voicePath_store2 = '.mat';
    path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
    save(path_store_voice, 'MSST_motion1', 'MSST_motion2');

    % motion_2
    [mappedD1, ~] = compute_mapping(squeeze(motion_2(1:89,:)'), 'PCA', 1);%上半段
    [mappedD2, ~] = compute_mapping(squeeze(motion_2(89:end,:)'), 'PCA', 1);%下半段
    mappedD1 = smooth(mappedD1,10);
    mappedD2 = smooth(mappedD2,10);
    mappedD1 = zscore(mappedD1);
    mappedD2 = zscore(mappedD2);
    if (max(mappedD1)<2)
    mappedD1 = -mappedD1;
    end
    if (max(mappedD2)<2)
    mappedD2 = -mappedD2;
    end
    MSST_motion1 = real(IMSST_W(mappedD1,40,10));
    MSST_motion2 = real(IMSST_W(mappedD2,40,10));
    MSST_motion1 = MSST_motion1(1:60,:);
    MSST_motion2 = MSST_motion2(1:60,:);
    % % save MSST figure
    % figure();
    % subplot(2,1,1);
    % imagesc(MSST_motion1);
    % subplot(2,1,2);
    % imagesc(MSST_motion2);
    % figureSavePath1 = strcat(participant,'\Figure\4motion_MSST\');
    % figureSavePath2  = 'motion2.jpg';
    % figureSavePath_1 = sprintf('%s%d%s',figureSavePath1, times_number, figureSavePath2);
    % saveas(gcf, figureSavePath_1);
    % close all
    voicePath_store0  = strcat(participant,'\MSSTFeature\distractMotion\',type2,'\MSSTFeature');
    voicePath_store2 = '.mat';
    path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
    save(path_store_voice, 'MSST_motion1', 'MSST_motion2');
end



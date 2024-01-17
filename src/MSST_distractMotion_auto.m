%%  Configuration
participant = 'P_GQY_side'; 
flag = 1; % 0) eat fetch 1) turnBack pickUp
if (flag == 0)
    motion_name = 'eatFetch';
else
    motion_name = 'turnbackPickup';
end
property = dir(strcat(participant,'\CollectedData_',motion_name));
types = [];
for i = 3:length(property)
    temp = property(i).name;
    temp(1:14) = [];
    types = [types string(temp)];
end

for t = 1:length(types)
    type = types(t);
    %type = "eatFetch_mute";
    path_voice2 = strcat(participant,'\CollectedData_',motion_name,'\CollectedData_',type,'\CollectedData\WAV\SingleWav');
    split_type = strsplit(type,'_'); 
    if (flag == 0) % 0) eat fetch 1) turnBack pickUp
        mat_save_path_1 = strcat(participant,'\MSSTFeature_distractMotion\',split_type(2),'\fetch');
        mkdir(mat_save_path_1);
        mat_save_path_2 = strcat(participant,'\MSSTFeature_distractMotion\',split_type(2),'\eat');
        mkdir(mat_save_path_2);
    else
        mat_save_path_1 = strcat(participant,'\MSSTFeature_distractMotion\',split_type(2),'\turn_back');
        mkdir(mat_save_path_1);
        mat_save_path_2 = strcat(participant,'\MSSTFeature_distractMotion\',split_type(2),'\pick_up');
        mkdir(mat_save_path_2);
    end
    accs = dir(strcat(participant,'\CollectedData_',motion_name,'\CollectedData_',type,'\CollectedData\BINS'));
    acc_count = length(accs) - 2;
    
    for j = 1:acc_count

        times_number = j;
        try
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

            % sum frequency
            % Min-max normalize
            output_columnSum = zscore(sum(output)); %列求和
            output_columnSum = smooth(output_columnSum(:), 10);
            cloumnSum_sort = sort(output_columnSum);
            cloumnSum_index = fix(length(cloumnSum_sort)*0.8);
            threshold = cloumnSum_sort(cloumnSum_index);
            peak_index = [];
            for i = 1:length(output_columnSum)
                if (output_columnSum(i) > threshold)
                    peak_index = [peak_index i];
                end
            end
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
            motion_1_front_index = fix(motion_1_front/(T(2)-T(1)));
            motion_1_rear_index = fix(motion_1_rear/(T(2)-T(1)));
            motion_2_front_index = fix(motion_2_front/(T(2)-T(1)));
            motion_2_rear_index = fix(motion_2_rear/(T(2)-T(1)));
            motion_1 = output(:,motion_1_front_index:motion_1_rear_index);
            motion_2 = output(:,motion_2_front_index:motion_2_rear_index);
            %% MSSTFeature
            % motion_1
            [mappedD1, ~] = compute_mapping(squeeze(motion_1(1:60,:)'), 'PCA', 1);%上半段
            [mappedD2, ~] = compute_mapping(squeeze(motion_1(60:end,:)'), 'PCA', 1);%下半段
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

            voicePath_store0  = strcat(mat_save_path_1,'\MSSTFeature');
            voicePath_store2 = '.mat';
            path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
            save(path_store_voice, 'MSST_motion1', 'MSST_motion2');

            % motion_2
            [mappedD1, ~] = compute_mapping(squeeze(motion_2(1:60,:)'), 'PCA', 1);%上半段
            [mappedD2, ~] = compute_mapping(squeeze(motion_2(60:end,:)'), 'PCA', 1);%下半段
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

            voicePath_store0  = strcat(mat_save_path_2,'\MSSTFeature');
            voicePath_store2 = '.mat';
            path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
            save(path_store_voice, 'MSST_motion1', 'MSST_motion2');
        catch
        end
    end
    
end







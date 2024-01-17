%%  Configuration
participant = 'P_Jason_side'; 
property = dir(strcat(participant,'\CollectedData_changeLane'));
types = [];
for i = 3:length(property)
    temp = property(i).name;
    temp(1:14) = [];
    types = [types string(temp)];
end

for t = 1:length(types)
    type = types(t);
    
    path_voice2 = strcat(participant,'\CollectedData_changeLane\CollectedData_',type,'\CollectedData\WAV\SingleWav');
    ImuPath1 = strcat(participant,'\AccAndGyr_changeLane\AccAndGyr_',type,'\AccAndGyr\AccAndGyr');
    split_type = strsplit(type,'_');   
    mat_save_path_L = strcat(participant,'\MSSTFeature_changeLane\',split_type(1),'L_',split_type(3),'\',split_type(2));
    mkdir(mat_save_path_L);
    mat_save_path_R = strcat(participant,'\MSSTFeature_changeLane\',split_type(1),'R_',split_type(3),'\',split_type(2));
    mkdir(mat_save_path_R);
    turn_start_offset = 1; % (default: +1)
    duration = 5;
    % auto times_number
    % current dir
    accs = dir(strcat(participant,'\CollectedData_changeLane\CollectedData_',type,'\CollectedData\WAV'));
    acc_count = length(accs) - 2;
    
    for j = 1:acc_count

        times_number = j;

        %%  changeable path
        path_voice3 = '.wav';
        path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
        ImuPath2  = '.xls';
        ImuPath = sprintf('%s%d%s',ImuPath1, times_number, ImuPath2);
        %%  verify
        if (isfile(path_voice)&&isfile(ImuPath)) == 0
            continue; 
        end
        %%  Read SingleWav
        try
            [x, Fs] = audioread(path_voice); 
            org_x = x;
            x = x(1*Fs:end-2*Fs);  %x = x(5*Fs:end-2*Fs);  
            org_time = (length(org_x) - 1) / Fs;
            LenData = length(x); % the length of wav
            time = (0 : LenData-1) / Fs; 
            shape = size(x);
            fftRealArray = zeros(shape)'; 
            LenBatch = 3584; % android microphone buffer size is 3584
            Amp = zeros(floor(LenData/LenBatch), 4096); % amp contains all the bins 2048

            %% Read Acc n Gyr data。
            gyr = readcell(ImuPath,'Sheet','Gyr');
            Fs_imu = 50;
            gyry = cell2mat(gyr(2:end,2));
            org_gyry_time = (length(gyry) - 1) / Fs_imu;
            gyry = gyry(2*Fs_imu:end-2*Fs_imu);  
            gyrlen = length(gyry);
            x2 = (0:gyrlen-1)/Fs_imu; % -2是因为第一行为空
            gyry = smooth(gyry(:),100)';

            if (abs(org_time - org_gyry_time) > 1)
                continue
            end
            
            if (org_time < 8)
                continue
            end

            gyry_cutted = gyry(1*Fs_imu:end-1*Fs_imu);
            
            % distinguish change lane R or L; turn time
            whichChangeLane = 0;  % 0: right右 1: left左
            turn_start = 0;
            for i=1:length(gyry_cutted)
                if (gyry_cutted(i)<=-0.02) % changeLaneR
                    turn_start = x2(i) + turn_start_offset; % time(second)
                    break;
                end
                if (gyry_cutted(i)>= 0.02) % changeLaneL
                    turn_start = x2(i) + turn_start_offset; % time(second)
                    whichChangeLane = 1;
                    break;
                end
            end
            % if (turn_start <= 3)
            %     return 
            % end
            if (turn_start - duration < 0)
                turn_start = duration;
            end

%             [S,F,T,P] = spectrogram(x, hanning(round(fix(23.1*0.01*Fs)/2)), round(fix(23.1*0.01*Fs)/2.5), fix(23.1*0.01*Fs*2.5), Fs);
%             % fetch 19850-20150 Hz in spectrogram
%             F_step = F(3) - F(2);
%             F_first_index = floor(19850/F_step);
%             F_end_index = floor(20150/F_step);
%             P = P(F_first_index:F_end_index,:);
%             output = 10*log10(abs(P));
%             figure();
%             % show spectrogram
%             subplot(2,1,1);
%             % imagesc(output);
%             imagesc(T,F(F_first_index:F_end_index),output); 
%             axis([0 org_time-4,-inf,inf]);
%             xlabel('Time (Seconds)'); ylabel('Hz');
%             % show gyry 
%             subplot(2,1,2);
%             plot(x2,gyry,'b'); % plot(x2,gyrx,'b');
%             xlabel('Time (s)');
%             ylabel('Gyro Y');
%             axis([0 org_time-4,-inf,inf]);
%             hold on;
%             plot([turn_start,turn_start],[min(gyry), max(gyry)]);
        
            %% MSSTFeature
            EsdProcess = [
                1 % Divde bins
            ];

            handle_esd = [
                1
                1 
                1
            ];

            for n = 1:(LenData/LenBatch)
                [z,p,k] = butter(9,19800/(Fs/2),'high');
                [sos_filter ,g] = zp2sos(z,p,k);
                fftRealArray = x( ((n-1)*3584 + 1) : (n*3584) ); % Extract Batch data.
                fftRealArray = sosfilt(sos_filter,fftRealArray) *g;
                % Regulate in amplitude
                for i = 1:LenBatch
                    fftRealArray(i) = fftRealArray(i) / 32768; 
                end
                %Haming Window
                for i = 1:(LenBatch/2) 
                    winval = 0.5 + 0.5 * cos( (pi*i) / (LenBatch/2));
                    if i > LenBatch/2  
                        winval = 0;
                    end
                    fftRealArray(LenBatch/2 + i) = fftRealArray(LenBatch/2 + i) * winval;
                    fftRealArray(LenBatch/2 + 1 - i) = fftRealArray(LenBatch/2 + 1 - i) * winval;
                end
                %fft
                result = fft(fftRealArray, 4096*2); 
                amp = abs(result);
                amp = amp(1:4096); 
                Amp(n,:) = amp' .* 1000;
            end
            [Amp_r, Amp_c] = size(Amp);
            EFs = (Amp_r -1) / ((LenData-1)/Fs);
            ETic = 0:1/EFs:(Amp_r-1) / EFs;
            f_bin = (0:Amp_r-1) * EFs / Amp_r;

            FeatureSafe = [];
            tfrlast = 0;
            if (EsdProcess(1) == 1)
                SubEsd = zeros(21,Amp_r);
                j = 1; 
                subplot_ind = 1;
                for i = 3705:1:3725
                        % SubEsd(j,:) =Amp(:, i);
                        if (handle_esd(2) == 1)
                            if(handle_esd(3) == 1)
                                % Esd = Amp(:, i)';
                                Esd = smooth(Amp(:, i),10)';
                            end
                            BGMK = 10;
                            UpdataRate = 0; % [0,1]
                            % calculate model.
                            BackGround = (1/BGMK) * sum(Esd(1:BGMK));
                            % old ESD
                            BackGround_old = 0;
                            % update model.
                            for i = BGMK:1:length(Esd)
                                UpdataRate = abs(Esd(i) - BackGround_old) / max(Esd(i), BackGround_old); % update rate
                                BackGround = (1 - UpdataRate) * BackGround + Esd(i) * UpdataRate; %update model
                                BackGround_old = Esd(i); 
                                Esd(i) = Esd(i) - BackGround; % calculate ESD /subtract the Background
                            end
%                         subplot(5,5,subplot_ind);plot(ETic , Esd );title('Denoise Signals');
%                         subplot_ind = subplot_ind + 1;
                        end
                        SubEsd(j,:) = Esd;
                        j = j+1;
                        tfr0 = IMSST_W(Esd(1*EFs:end)',40,10);
                        [tfr0_f, tfr0_t] = size(tfr0);
                        ratio = tfr0_t / length(Esd);
                        % figure;
                        % imagesc(ETic,EFs,abs(tfr0));
                        FeatureSafe = cat(3, FeatureSafe,abs(tfr0));
                end
                EsdData = SubEsd(1:j-1,:); %Extract Bins
            end

            % cutted 
            [f_size, t_szie, bins] = size(FeatureSafe);
            if (f_size > 60)
                f_size = 60;
            end
            FeatureSafeNew = FeatureSafe(1:f_size,:,:); % (frequency, time, bin) % f: 1-60 %　EFs*2:end
            if (turn_start-duration <= 0.1)
                MSSTFeature = FeatureSafeNew(:,1:fix(EFs/ratio)*turn_start,:); % t 
            else
                MSSTFeature = FeatureSafeNew(:,fix(EFs/ratio)*(turn_start-duration):fix(EFs/ratio)*turn_start,:); % t 
            end
           
            %% save data
            if (whichChangeLane) % changeLaneL
                voicePath_store0  = strcat(mat_save_path_L,'\MSSTFeature');
            else 
                voicePath_store0  = strcat(mat_save_path_R,'\MSSTFeature');
            end
            voicePath_store2 = '.mat';
            path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
            save(path_store_voice, 'ETic','EFs','MSSTFeature', 'turn_start');
            
        catch
        end
    end
    
end




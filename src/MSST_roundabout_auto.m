%%  Configuration
participant = 'P_GQY_side'; 
property = dir(strcat(participant,'\CollectedData_roundabout'));
types = [];
for i = 3:length(property)
    temp = property(i).name;
    temp(1:14) = [];
    types = [types string(temp)];
end

for t = 1:length(types)
    type = types(t);
    
    path_voice2 = strcat(participant,'\CollectedData_roundabout\CollectedData_',type,'\CollectedData\WAV\SingleWav');
    ImuPath1 = strcat(participant,'\AccAndGyr_roundabout\AccAndGyr_',type,'\AccAndGyr\AccAndGyr');
    split_type = strsplit(type,'_');   
    mat_save_path = strcat(participant,'\MSSTFeature_roundabout\',split_type(1),'_',split_type(3),'\',split_type(2));
    mkdir(mat_save_path);
    voicePath_store0  = strcat(mat_save_path,'\MSSTFeature_roundabout');
    turn_start_offset = 4; % (default: +1)
    duration = 5;
    % auto times_number
    % current dir
    accs = dir(strcat(participant,'\CollectedData_roundabout\CollectedData_',type,'\CollectedData\WAV'));
    acc_count = length(accs) - 2;
    
    for j = 1:acc_count
    
        times_number = j;
        try
            %%  changeable path
            path_voice2 = strcat(participant,"\CollectedData_roundabout\CollectedData_",type,"\CollectedData\WAV\SingleWav");
            BinsPath1 = strcat(participant,"\CollectedData_roundabout\CollectedData_",type,"\CollectedData\BINS\TEST_BINS");
            ImuPath1 = strcat(participant,"\AccAndGyr_roundabout\AccAndGyr_",type,"\AccAndGyr\AccAndGyr");

            %%  Read SingleWav
            path_voice3 = '.wav';
            path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
            BinsPath2  = '.csv';
            BinsPath = sprintf('%s%d%s',BinsPath1, times_number, BinsPath2);
            ImuPath2  = '.xls';
            ImuPath = sprintf('%s%d%s',ImuPath1, times_number, ImuPath2);
            if (isfile(path_voice)) == 0
                continue 
            end
            if (isfile(BinsPath)) == 0
                continue 
            end
            if (isfile(ImuPath)) == 0
                continue 
            end
            %%  Read SingleWav
            [x, Fs] = audioread(path_voice); 
            org_x = x;
            x = x(1.5*Fs:end-1.5*Fs);  %x = x(5*Fs:end-2*Fs);  
            org_time = (length(org_x) - 1) / Fs;
            LenData = length(x); % the length of wav
            time = (0 : LenData-1) / Fs; 
            shape = size(x);
            fftRealArray = zeros(shape)'; 
            LenBatch = 3584; % android microphone buffer size is 3584
            Amp = zeros(floor(LenData/LenBatch), 4096); % amp contains all the bins 2048

            %%  Read time 
            start_bins_time_all = csvread(BinsPath,1,0);
            x_start_time = start_bins_time_all(1,1);
            date = datestr((x_start_time+28800000+time_offset)/86400000 + datenum(1970,1,1),31);

            %% Read Acc n Gyr data。
            gyr = readcell(ImuPath,'Sheet','Gyr');
            Fs_imu = 50;
            gyry = cell2mat(gyr(2:end,2));
            org_gyry = gyry;
            org_gyry_time = (length(gyry) - 1) / Fs_imu;
            gyry = gyry(1.5*Fs_imu:end-1.5*Fs_imu);  
            gyrlen = length(gyry);
            x2 = (0:gyrlen-1)/Fs_imu; % -2是因为第一行为空
            gyry = smooth(gyry(:),100)';

            if (abs(org_time - org_gyry_time) > 1)
                continue
            end
            if (org_time < 8 || org_gyry_time < 8)
                continue
            end

            gyry_cutted = gyry(1*Fs_imu:end-1*Fs_imu);

            %% Find turnStart and turnEnd point
            turn_start = 0;
            for i=1:length(gyry_cutted)
                if (gyry_cutted(i)>=0.3) % turnR - 0.1
                    turn_start = x2(i) + turn_start_offset; % time(second)
                    break;
                end
            end 
            if (turn_start==0)
                continue 
            end
            if (turn_start - duration < 0)
                turn_start = 1 + duration;
            end

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
                        subplot(5,5,subplot_ind);plot(ETic , Esd );title('Denoise Signals');
                        subplot_ind = subplot_ind + 1;
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
            voicePath_store2 = '.mat';
            path_store_voice = sprintf('%s%d%s',voicePath_store0,times_number,voicePath_store2);
            save(path_store_voice, 'ETic','EFs','MSSTFeature', 'turn_start');
        catch
        end
    end
end





close all
%%  Configuration
times_number = 14;
participant = 'P_GQY'; 
type = 'turnR_look';
road_condition = 1; % 1) turnR右 (-0.15)  2) turnL左 (+0.15)
turn_start_offset = 1; % (default: +1)
no_time_record = 0;
duration = 5;
showSpectrogram = 1;
showCuttedMSSTFeature = 1;
showFullMSSTFeature = 1;
%%  changeable path
if (no_time_record)
    path_voice2 = strcat(participant,'\CollectedData\CollectedData_',type,'\CollectedData\WAV\SingleWav (');
    ImuPath1 = strcat(participant,'\AccAndGyr\AccAndGyr_',type,'\AccAndGyr\AccAndGyr (');
    path_voice3 = ').wav';
    path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
    ImuPath2  = ').xls';
    ImuPath = sprintf('%s%d%s',ImuPath1, times_number, ImuPath2);
else
    path_voice2 = strcat(participant,'\CollectedData\CollectedData_',type,'\CollectedData\WAV\SingleWav');
    ImuPath1 = strcat(participant,'\AccAndGyr\AccAndGyr_',type,'\AccAndGyr\AccAndGyr');
    path_voice3 = '.wav';
    path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
    ImuPath2  = '.xls';
    ImuPath = sprintf('%s%d%s',ImuPath1, times_number, ImuPath2);
end
%%  verify
if (isfile(path_voice)&&isfile(ImuPath)) == 0
    return 
end
%%  Read SingleWav
[x, Fs] = audioread(path_voice); 
org_time = (length(x) - 1) / Fs;
x = x(2*Fs:end-2*Fs); 
LenData = length(x); 
time = (0 : LenData-1) / Fs; 
shape = size(x);
fftRealArray = zeros(shape)'; 
LenBatch = 3584; 
Amp = zeros(floor(LenData/LenBatch), 4096); 
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
    return
end
if (org_time < 8 || org_gyry_time < 8)
    return
end

%% Find turnStart and turnEnd point
% 1) turnR 2) turnL 
turn_start = 0;
switch road_condition
    case 1 % turnR
        for i=1:length(gyry)
            if (gyry(i)<=-0.2) % turnR - 0.1
                turn_start = x2(i) + turn_start_offset; % time(second)
                break;
            end
        end 
    case 2 % turnL
        for i=1:length(gyry)
            if (gyry(i)>=0.2) % turnL 0.1
                turn_start = x2(i) + turn_start_offset; % time(second)
                break;
            end
        end 
end
%%  STFT
if (showSpectrogram)
    [S,F,T,P] = spectrogram(x, hanning(round(fix(23.1*0.01*Fs)/2)), round(fix(23.1*0.01*Fs)/2.5), fix(23.1*0.01*Fs*2.5), Fs);
    % fetch 19850-20150 Hz in spectrogram
    F_step = F(3) - F(2);
    F_first_index = floor(19800/F_step);
    F_end_index = floor(20200/F_step);
    P = P(F_first_index:F_end_index,:);
    output = 10*log10(abs(P));
    % show spectrogram
    subplot(2,1,1);
    colormap Hot;
    clims = [-120 -60];
    imagesc(T,F(F_first_index:F_end_index),output,clims); 
    axis([0 org_time-4,-inf,inf]);
    xlabel('Time (Seconds)'); ylabel('Hz');
    % show gyry 
    subplot(2,1,2);
    plot(x2,gyry,'b'); % plot(x2,gyrx,'b');
    xlabel('Time (s)');
    ylabel('Gyro Y');
    axis([0 org_time-4,-inf,inf]);
    hold on;
    plot([turn_start,turn_start],[min(gyry), max(gyry)]);
end

if (turn_start==0)
    return 
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
figure();
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
            subplot(5,5,subplot_ind);plot(ETic(12:end), Esd(12:end));%title('Denoise Signals');
            subplot_ind = subplot_ind + 1;
            end
            SubEsd(j,:) = Esd;
            j = j+1;
            tfr0 = IMSST_W(Esd',40,10);
            [tfr0_f, tfr0_t] = size(tfr0);
            ratio = tfr0_t / length(Esd);
%             figure;
%             imagesc(ETic,EFs,abs(tfr0));
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
    start_get = 1; end_get = fix(EFs/ratio)*turn_start;
else
    start_get = fix(EFs/ratio)*(turn_start-duration); end_get = fix(EFs/ratio)*turn_start;
end
MSSTFeature = FeatureSafeNew(:,start_get:end_get,:);

MSSTs= EFs/ratio;
MSSTic = 0:1/MSSTs:(tfr0_t-1)/MSSTs;

if (showCuttedMSSTFeature)
    figure();
    %for i = 1:1:21
    %subplot(5,5,i);
    % clims = [0.0001 0.1];
    colormap Hot;
    imagesc(MSSTic(start_get:end_get),1:30,MSSTFeature(1:30,:,13));
    xlabel('Time / s');
    ylabel('Fre / Hz');
    %end
end


if (showFullMSSTFeature)
    for i = 1:5:21
        figure();
        clims = [0.0001 0.1];
        colormap Hot;
        imagesc(MSSTic(140:end),1:30,FeatureSafe(1:30,140:end,i),clims);    
        % set(gca,'FontSize',4); % 刻度
        title(strcat('Bins',string(i)));
        xlabel('Time / s'); % 'FontSize',4
        ylabel('Fre / Hz');
    end
end




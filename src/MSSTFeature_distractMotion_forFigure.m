close all
%%  Configuration
participant = 'P_GQY'; 
type = 'distractMotion_2';
times_number = 40;
duration = 5;
showSpectrogram = 0;
%%  changeable path
path_voice2 = strcat(participant,'\CollectedData\CollectedData_',type,'\CollectedData\WAV\SingleWav');
%%  verify
path_voice3 = '.wav';
path_voice = sprintf('%s%d%s',path_voice2, times_number, path_voice3);
if (isfile(path_voice)) == 0
    return 
end
%%  Read SingleWav
[x, Fs] = audioread(path_voice); 
x = x(2*Fs:end-2*Fs);  %x = x(5*Fs:end-2*Fs);  
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
F_first_index = floor(19800/F_step);
F_end_index = floor(20200/F_step);
P = P(F_first_index:F_end_index,:);
output = 10*log10(abs(P));

% show spectrogram
if (showSpectrogram)
    figure();
    imagesc(T,F(F_first_index:F_end_index),output); 
    xlabel('Time (Seconds)'); ylabel('Hz');
end


%% MSSTFeature

[mappedD1, ~] = compute_mapping(squeeze(output(1:120,:)'), 'PCA', 1);
[mappedD2, ~] = compute_mapping(squeeze(output(120:end,:)'), 'PCA', 1);
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
figure();
colormap Hot;
subplot(2,1,1);
clims = [1 60];
imagesc(T,1:60,MSST_motion1,clims);
title('Bins1');
xlabel('Time / s');
ylabel('Fre / Hz');
subplot(2,1,2);
imagesc(T,1:60,MSST_motion2,clims);
title('Bins2');
xlabel('Time / s');
ylabel('Fre / Hz');






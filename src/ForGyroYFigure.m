function[gyry, x2] = ForGyroYFigure(participant,type,times_number)
    ImuPath1 = strcat(participant,'\AccAndGyr\AccAndGyr_',type,'\AccAndGyr\AccAndGyr');
    ImuPath2  = '.xls';
    ImuPath = sprintf('%s%d%s',ImuPath1, times_number, ImuPath2);
    gyr = readcell(ImuPath,'Sheet','Gyr');
    Fs_imu = 50;
    gyry = cell2mat(gyr(2:end,2));
    gyry = gyry(2*Fs_imu:end-2*Fs_imu);  
    gyrlen = length(gyry);
    x2 = (0:gyrlen-1)/Fs_imu;
    gyry = smooth(gyry(:),100)';
end
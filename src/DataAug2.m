function DataAug2(participant,flag)

%    participant = 'Ran';
    duration=5;

%    flag = 0; % 0) eat 1）fetch 2) turnBack 3）pickUp

    switch flag
        case 0
        motion_name = 'eat';
        case 1
        motion_name = 'fetch';
        case 2
        motion_name = 'pick_up';
        case 3
        motion_name = 'turn_back';
    end

    mat_read_path1 = strcat('P_',participant,'/MSSTFeature/distractMotion/',motion_name,'/MSSTFeature');
    mat_save_path1 = strcat('P_',participant,'/MSSTFeature_new/distractMotion/',motion_name);
    mkdir(mat_save_path1);
    accs=dir(strcat('P_',participant,'/MSSTFeature/distractMotion/',motion_name));
    acc_count = length(accs)-2;

     % 遍历当前目录
     augNumber = 0;
    for j = 1:acc_count
        times_number = j;
        %read
        mat_read_path = sprintf('%s%d%s',mat_read_path1, times_number,'.mat');
        if (isfile(mat_read_path)) == 0
            continue
        end
        onedata=load(mat_read_path);
        databox.MSST_motion1=onedata.MSST_motion1;
        % 遍历剩下的文件
        for i = 1:acc_count
            i_times_number = i;
            i_mat_read_path = sprintf('%s%d%s',mat_read_path1, i_times_number,'.mat');
            if (i_times_number == j)
                continue
            end
            if (isfile(mat_read_path)) == 0
                continue
            end
            tmpdata=load(mat_read_path);
            databox.MSST_motion2=tmpdata.MSST_motion2;

            dataAug.MSST_motion1= databox.MSST_motion1;
            dataAug.MSST_motion2= databox.MSST_motion2;

            MSST_motion1=databox.MSST_motion1;
            MSST_motion2=databox.MSST_motion2;

            augNumber = augNumber +1;
            mat_save_path = sprintf('%s%s%d%s',mat_save_path1, '/MSSTFeature', augNumber,'.mat');
%            save(mat_save_path, 'dataAug');
            save(mat_save_path, 'MSST_motion1','MSST_motion2');
        end

    end
end

    
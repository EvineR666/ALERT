function DataAug_distract_gnoise(person,root)
    participant = person;
    for flag = 0:3   % 0) eat 1）fetch 2) turnBack 3）pickUp
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
%        mat_read_path1 = strcat('P_',participant,'/MSSTFeature/distractMotion/',motion_name,'/MSSTFeature');
        if participant =="P_GQY"
            mat_read_path1 = strcat(root,'/distractMotion/',motion_name,'/2_MSSTFeature');
        else
            mat_read_path1 = strcat(root,'/distractMotion/',motion_name,'/MSSTFeature');
        end
        mat_save_path1 = strcat(participant,'/MSSTFeature_new_gnoise/distractMotion','/',motion_name);
        mkdir(mat_save_path1);
%        accs=dir(strcat('P_',participant,'/MSSTFeature/distractMotion/',motion_name));
%        acc_count = length(accs)-2;
%        mat_read_path1 = strcat(participant,'\MSSTFeature\distractMotion\',motion_name,'\MSSTFeature');
%        mat_save_path1 = strcat('DataAug\',participant,'\MSSTFeature\distractMotion\',motion_name);
%        mkdir(mat_save_path1);
%        accs=dir(strcat(participant,'\MSSTFeature\distractMotion\',motion_name));
%        acc_count = length(accs)-2;

         % 遍历当前目录
         augNumber = 0;
        for j = 1:120
            times_number = j;
            read_path = sprintf('%s%d%s',mat_read_path1, times_number,'.mat');
            if (isfile(read_path)) == 0
                continue
            end
            % 遍历剩下的文件
            for i = 1:120
                i_times_number = i;
                mat_read_path = sprintf('%s%d%s',mat_read_path1, i_times_number,'.mat');
                if (i_times_number == j)
                    continue
                end
                if (isfile(mat_read_path)) == 0
                    continue
                end
                onedata=load(mat_read_path);
                p=rand;
                gnoise=wgn1(60,217,p);
                maxnoise=max(max(gnoise));
                minnoise=min(min(gnoise));
                gnoise=(gnoise-minnoise)*0.01/(maxnoise-minnoise);
                MSST_motion1=onedata.MSST_motion1(:,1:217)+gnoise;
                MSST_motion2=onedata.MSST_motion2(:,1:217)+gnoise;

                augNumber = augNumber +1;
                mat_save_path = sprintf('%s%s%d%s',mat_save_path1, '/MSSTFeature', augNumber,'.mat');
                save(mat_save_path, 'MSST_motion1','MSST_motion2');
            end

        end

    end
end

    
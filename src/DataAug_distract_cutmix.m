function DataAug_distract_cutmix(person,root)
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
        mat_save_path1 = strcat(participant,'/MSSTFeature_new_cutmix/distractMotion','/',motion_name);
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
            %read
            mat_read_path = sprintf('%s%d%s',mat_read_path1, times_number,'.mat');
            if (isfile(mat_read_path)) == 0
                continue
            end
            onedata=load(mat_read_path);
            cut_MSST_motion1=onedata.MSST_motion1(:,1:72);%60,217
            cut_MSST_motion2=onedata.MSST_motion2(:,1:72);
            % 遍历剩下的文件
            for i = 1:120
                i_times_number = i;
                i_mat_read_path = sprintf('%s%d%s',mat_read_path1, i_times_number,'.mat');
                if (i_times_number == j)
                    continue
                end
                if (isfile(i_mat_read_path)) == 0
                    continue
                end
                tmpdata=load(i_mat_read_path);
                tmpdata.MSST_motion1(:,1:72)=cut_MSST_motion1;
                tmpdata.MSST_motion2(:,1:72)= cut_MSST_motion2;

                augNumber = augNumber +1;
                mat_save_path = sprintf('%s%s%d%s',mat_save_path1, '/MSSTFeature', augNumber,'.mat');
                save(mat_save_path, 'tmpdata');
            end

        end

    end
end

    
function DataAug_road(person,root,motion_name)
%    participant='P_molly';
    participant=person;
    kinds=["look","no"];
    for kind=kinds
%    for flag = 1
%        switch flag
%            case 0
%            motion_name = 'changeLaneL';
%            case 1
%            motion_name = 'changeLaneR';
%            case 2
%            motion_name = 'roundabout';
%            case 3
%            motion_name = 'turnL';
%            case 4
%            motion_name = 'turnR';
%        end

        read_path1 = strcat(root,'/',motion_name,'/',kind,'/MSSTFeature');
        save_path1 = strcat(participant,'/MSSTFeature_new/peak_',motion_name,'/',kind);
        mkdir(save_path1);
%             accs=dir(strcat(participant,'\MSSTFeature\',motion_name,'\look'));
    %         acc_count = length(accs)-2;

         % 遍历当前目录
        augNumber = 0;
        for j = 1:100
            times_number = j;
            %read
            read_path = sprintf('%s%d%s',read_path1, times_number,'.mat');
            if (isfile(read_path)) == 0
                continue
            end
            MSSTFeature1=load(read_path);
            [a,b,c]=size(MSSTFeature1.MSSTFeature);
            MSSTFeature1.MSSTFeature=[MSSTFeature1.MSSTFeature,zeros(a,66-b,21);zeros(60-a,66,21)];
            data_bin16=MSSTFeature1.MSSTFeature(:,:,16);
            % 遍历剩下的文件
            for i = 1:100
                i_times_number = i;
                i_read_path = sprintf('%s%d%s',read_path1, i_times_number,'.mat');
                if (i_times_number == j)
                    continue
                end
                if (isfile(i_read_path)) == 0
                    continue
                end
                MSSTFeature=load(i_read_path);
                [a,b,c]=size(MSSTFeature.MSSTFeature);
                MSSTFeature.MSSTFeature=[MSSTFeature.MSSTFeature,zeros(a,66-b,21);zeros(60-a,66,21)];
                MSSTFeature.MSSTFeature(:,:,16)=data_bin16;

                augNumber = augNumber +1;
                save_path = sprintf('%s%s%d%s',save_path1, '/MSSTFeature', augNumber,'.mat');
%                 turn_start=MSSTFeature.turn_start;
%                 EFs=MSSTFeature.EFs;
%                 ETic=MSSTFeature.ETic;
                turn_start=0;
                EFs=0;
                ETic=0;
                MSSTFeature=MSSTFeature.MSSTFeature;
                save(save_path, 'turn_start','EFs','ETic','MSSTFeature');
            end

        end

    end
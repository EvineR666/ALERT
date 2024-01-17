function DataAug_road_gnoise(person,root,motion_name)
%    participant='P_molly';
    participant=person;
    kinds=["look","no"];
    for kind=kinds
    size_noise=0.01;


        read_path1 = strcat(root,'/',motion_name,'/',kind,'/MSSTFeature');
        save_path1 = strcat(participant,'/MSSTFeature_new_gnoise/',motion_name,'/',kind);
        mkdir(save_path1);


         % 遍历当前目录
        augNumber = 0;
        for j = 1:100
            times_number = j;
            read_path = sprintf('%s%d%s',read_path1, times_number,'.mat');
            if (isfile(read_path)) == 0
                continue
            end
            MSSTFeature=load(read_path);
            [a,b,~]=size(MSSTFeature.MSSTFeature);
            MSSTFeature.MSSTFeature=[MSSTFeature.MSSTFeature,zeros(a,66-b,21);zeros(60-a,66,21)];
            for i=1:100
                i_times_number = i;
                i_read_path = sprintf('%s%d%s',read_path1, i_times_number,'.mat');
                MSSTFeature=load(read_path);
                [a,b,~]=size(MSSTFeature.MSSTFeature);
                MSSTFeature.MSSTFeature=[MSSTFeature.MSSTFeature,zeros(a,66-b,21);zeros(60-a,66,21)];
                if (i_times_number == j)
                    continue
                end
                if (isfile(i_read_path)) == 0
                    continue
                end
                p=rand;
                gnoise=wgn1(60,66,p);
                maxnoise=max(max(gnoise));
                minnoise=min(min(gnoise));
                gnoise=(gnoise-minnoise)*size_noise/(maxnoise-minnoise);

                turn_start=MSSTFeature.turn_start;
                EFs=MSSTFeature.EFs;
                ETic=MSSTFeature.ETic;
                MSSTFeature=MSSTFeature.MSSTFeature+gnoise;
                augNumber = augNumber+1;
                save_path = sprintf('%s%s%d%s',save_path1, '/MSSTFeature', augNumber,'.mat');
                save(save_path, 'turn_start','EFs','ETic','MSSTFeature');
            end
        end

    end
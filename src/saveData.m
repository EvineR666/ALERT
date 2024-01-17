
% save MSST_turnHead_changeLane

% for i = 1:70
%     try
%         % MSST_distractMotion(i,'fetch','eat');
%         MSST_distractMotion(i,'turn_back','pick_up');
%         % MSST_distractMoiton(i)
%     catch
%     end
% end

% for i = 1:30
%     MSST_turn(i);
% end

% for i = 1:70
%     try
%         MSST_turn(i);
%     catch
%     end
% end

% for j = 1:70
%     try
%         MSST_distractMotion(j,'fetch','eat');
%     catch
%     end
% end

% for j = 2:2:90
%     try
%         MSST_distractMotion(j,'turn_back','pick_up')
%     catch
%     end
% end



%% save distract motion

% 分心動作: 單數: 1) fetch 2) eat  雙數: 1) turn back 2) pick up
% for j = 1:2:84
%     try
%         MSST_distractMotion(j,'fetch','eat')
%     catch
%     end
% end
% 
% for j = 2:2:84
%     try
%         MSST_distractMotion(j,'turn_back','pick_up')
%     catch
%     end
% end


%% ====================   ForGyroYFigure  ==============================

% each participant
participants = ["P_Finesse", "P_molly", "P_Samishikude", "P_陈昌忻", "P_然", "P_熊宇轩"];
road_type = "roundabout_no";
figureSavePath2  = '.jpg';
mat_save_path = strcat("all_gyry\",road_type,"\");
mkdir(mat_save_path);
for i = 1:length(participants)
    for j = 1:120
        try
            [gyry, ~] = ForGyroYFigure(participants(i),road_type,j);
            close all;
            figure();
            plot(gyry);
            figureSavePath_1 = strcat(mat_save_path,participants(i),"_",string(j), figureSavePath2);
            saveas(gcf, figureSavePath_1);
        catch
        end
    end
end
% each sample

% [changeLaneL_gyry, changeLaneL_x2] = ForGyroYFigure('P_GQY','changeLaneL_look',12);
% [changeLaneR_gyry, changeLaneR_x2] = ForGyroYFigure('P_GQY','changeLaneR_look',11);
% [turnL_gyry, turnL_x2] = ForGyroYFigure('P_GQY','turnL_look',17);
% [turnR_gyry, turnR_x2]  = ForGyroYFigure('P_GQY','turnR_look',16);
% [roundabout_gyry, roundabout_x2] = ForGyroYFigure('P_GQY','roundabout_look',10);
% 
% figure();
% subplot(2,3,1);
% plot(changeLaneL_x2,changeLaneL_gyry,'b');
% axis([-inf, inf -0.5 0.5]);
% title('Leftward Lane Change');
% xlabel('Time (s)');
% ylabel('Gyro Y');
% 
% subplot(2,3,2);
% plot(changeLaneR_x2,changeLaneR_gyry,'b');
% axis([-inf, inf -0.5 0.5]);
% title('Rightward Lane Change');
% xlabel('Time (s)');
% ylabel('Gyro Y');
% 
% subplot(2,3,3);
% plot(turnL_x2,turnL_gyry,'b');
% axis([-inf, inf -0.5 0.5]);
% title('Turn left');
% xlabel('Time (s)');
% ylabel('Gyro Y');
% 
% subplot(2,3,4);
% plot(turnR_x2,turnR_gyry,'b');
% axis([-inf, inf -0.5 0.5]);
% title('Turn right');
% xlabel('Time (s)');
% ylabel('Gyro Y');
% 
% subplot(2,3,5);
% plot(roundabout_x2,roundabout_gyry,'b');
% axis([-inf, inf -0.5 0.5]);
% title('Roundabout');
% xlabel('Time (s)');
% ylabel('Gyro Y');
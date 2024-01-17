#persons=(Jason GQY Finesse 然 Samishikude molly 陈昌忻 熊宇轩)
#persons=(molly)
persons=(陈昌忻)
#road_types=(distractMotion_LBom distractMotion_mute distractMotion_LUp distractMotion_RBom)
#road_types=(changelaneL_LBom changelaneL_mute changelaneR_LBom changelaneR_mute changelaneL_LUp changelaneL_RBom changelaneR_LUp changelaneR_RBom)


#for person in ${persons[@]}
#do
#  gpu_id=`expr ${t} % 2`
#  echo "gpu_id:${gpu_id}"
#  #判断日志文件夹是否存在
#  if [ ! -d "Log/${person}" ];then
#    mkdir Log/${person}
#  fi
#
#  nohup python train_distract.py ${person} ${gpu_id} > Log/${person}/result.txt &
#
#  sleep 10
#
#  t=`expr ${t} + 1`
#done

#road_types=(turnL_LBom turnL_mute turnR_LBom turnR_mute turnL_LUp turnL_RBom turnR_LUp turnR_RBom)
#road_types=(roundabout_LBom roundabout_mute roundabout_LUp roundabout_RBom)
#road_types=(offpeak_changeLaneL offpeak_changeLaneR offpeak_roundabout offpeak_turnR offpeak_turnL)
#road_types=(peak_changeLaneR peak_roundabout peak_turnL peak_turnR peak_changeLaneL )
road_types=(distractMotion)
persons=(GQY)
t=0
for person in ${persons[@]}
do
  for road_type in ${road_types[@]}
  do
    gpu_id=`expr ${t} % 2`
    echo "gpu_id:${gpu_id}"
    #判断日志文件夹是否存在
    if [ ! -d "Log/${person}" ];then
      mkdir Log/${person}
    fi

    nohup python train_winSize_new.py ${person} ${gpu_id} ${road_type}> Log/${person}/${road_type}_result.txt  2>&1 &
#    nohup python train.py ${person} ${gpu_id} ${road_type} > Log/${person}/${road_type}_result.txt &

    sleep 10

    t=`expr ${t} + 1`
  done
done

#road_types=(changeLaneL changeLaneR roundabout turnL turnR)#road_types=(distractMotion_LBom distractMotion_mute distractMotion_LUp distractMotion_RBom)

#road_types=(distractMotion)
#model_types=(1 2 3)
#person=(molly)
#for road_type in ${road_types[@]}
#do
#  for model_type in ${model_types[@]}
#  do
#    gpu_id=`expr ${t} % 2`
#    echo "gpu_id:${gpu_id}"
#    #判断日志文件夹是否存在
#    if [ ! -d "Log/${person}" ];then
#      mkdir Log/${person}
#    fi
#
##    nohup python train_distract.py ${person} ${gpu_id} ${road_type}> Log/${person}/${road_type}_result.txt &
#    nohup python train_distract.py ${person} ${gpu_id} ${road_type} ${model_type} > Log/${person}/${model_type}_${road_type}_result.txt &
#
#    sleep 10
#
#    t=`expr ${t} + 1`
#  done
#done

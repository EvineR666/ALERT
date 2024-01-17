person='P_GQY';
root='P_GQY/MSSTFeature';
motions=["roundabout","turnL","turnR"];
% motions=["turnL","turnR"];
for motion_name=motions
    DataAug_road(person,root,motion_name);
end
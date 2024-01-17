persons=["P_GQY","P_Finesse","P_Jason","P_molly","P_Samishikude","P_然","P_陈昌忻","P_熊宇轩"];
motions=["turnL","turnR","roundabout","changeLaneL","changeLaneR","distractMotion"];
for person=persons
    root=strcat(person,'/MSSTFeature');
    DataAug_distract_gnoise(person,root);
    DataAug_distract_cutmix(person,root);
    for motion_name=motions
        DataAug_road_gnoise(person,root,motion_name);
        DataAug_road_cutmix(person,root,motion_name);
    end
end
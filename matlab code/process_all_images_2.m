clc;
clear all;
close all;

nn = 11468;

dir_folder = 'G:\Program Files\Work\deep_learning\whale_images\imgs_head';
save_folder = 'G:\Program Files\Work\deep_learning\whale_images\imgs_head_bluechannel';


parfor i=0:0
    if i==7489
        continue;
    end
    rawimg = imread(strcat(dir_folder, '\w_', num2str(i), '.jpg'));
    blue_channel = rawimg(:,:,3);
    imwrite(blue_channel, strcat(save_folder, '\w_', num2str(i), '.jpg'), 'jpg');
end
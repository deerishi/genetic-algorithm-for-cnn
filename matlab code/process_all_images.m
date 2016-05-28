clc;
clear all;
close all;

nn = 11468;

dir_folder = 'G:\Program Files\Work\deep_learning\whale_images\imgs';
save_folder = 'G:\Program Files\Work\deep_learning\whale_images\all_blue_channel';
filelist = ls(dir_folder);
filelist = filelist(3:length(filelist),:);

filenum = length(filelist);

parfor i=11000:11468
    if i==7489
        continue;
    end
    rawimg = imread(strcat(dir_folder, '\w_', num2str(i), '.jpg'));
    whale_head = detect_whale_face(rawimg);
    imwrite(whale_head, strcat(save_folder, '\w_', num2str(i), '.jpg'), 'jpg');
end

clc;
clear all;
close all;

nn = 12;

dir_folder = 'D:\Xinran\Matlab Program\STAT946\imgs_subset';
filelist = ls(dir_folder);
filelist = filelist(3:length(filelist),:);

filenum = length(filelist);
sublist = 500*rand(nn, 1);
sublist = floor(sublist);

figure;
for i=1:nn
    rawimg = imread(strcat(dir_folder, '\w_', num2str(sublist(i)), '.jpg'));
    whale_head = detect_whale_face(rawimg);
    subplot(3, 4, i);
    imshow(whale_head);
    axis off;
    %colormap(gray(255));
end

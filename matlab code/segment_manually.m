
clc;
close all;
clear all;

detect_folder = 'G:\Program Files\Work\deep_learning\detect_all';
manual_folder = 'G:\Program Files\Work\deep_learning\manual_all';
dir_folder = 'G:\Program Files\Work\deep_learning\imgs';

filelist = ls(detect_folder);
filelist = filelist(3:length(filelist),:);
filenum = length(filelist);

numlist = zeros(filenum, 1);
for i=1:filenum
    tempname = filelist(i, :);
    num = tempname(3:(strfind(tempname,'.jpg')-1));
    numlist(i) = str2num(num);
end

for i=7490:11468
    if(find(numlist==i))
        continue;
    end
    
    rawimg = imread(strcat(dir_folder, '\w_', num2str(i), '.jpg'));
    [height, width, iscolor] = size(rawimg);
    if width>250
        img = imresize(rawimg, 0.1);
    else
        img = imresize(rawimg, 0.2);
    end
    [height, width, iscolor] = size(img);
    
    figure;
    imshow(img);
    title(num2str(i));
    
    [ex,ey] = ginput(1);
    xi=round(ex);
    yi=round(ey);
    
    boxhfedge = round(width/10);
    upleftx = max(1, (xi-boxhfedge));
    uplefty = max(1, (yi-boxhfedge));
    btrightx = min(width, (xi+boxhfedge));
    btrighty = min(height, (yi+boxhfedge));
    
    outputimg = img;
    outputimg(uplefty:uplefty+1, upleftx:btrightx, 1) = 255;
    outputimg(uplefty:uplefty+1, upleftx:btrightx, 2) = 0;
    outputimg(uplefty:uplefty+1, upleftx:btrightx, 3) = 0;
    outputimg(btrighty:btrighty+1, upleftx:btrightx, 1) = 255;
    outputimg(btrighty:btrighty+1, upleftx:btrightx, 2) = 0;
    outputimg(btrighty:btrighty+1, upleftx:btrightx, 3) = 0;
    outputimg(uplefty:btrighty, upleftx:upleftx+1, 1) = 255;
    outputimg(uplefty:btrighty, upleftx:upleftx+1, 2) = 0;
    outputimg(uplefty:btrighty, upleftx:upleftx+1, 3) = 0;
    outputimg(uplefty:btrighty, btrightx:btrightx+1, 1) = 255;
    outputimg(uplefty:btrighty, btrightx:btrightx+1, 2) = 0;
    outputimg(uplefty:btrighty, btrightx:btrightx+1, 3) = 0;
    
    imshow(outputimg);
    pause(0.5);
    
    close all;
    imwrite(outputimg, strcat(manual_folder, '\w_', num2str(i), '.jpg'), 'jpg');
end

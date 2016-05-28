
clc;
close all;
clear all;

dir_folder = 'G:\Program Files\Work\deep_learning\whale_images\imgs_head_bluechannel';
[NUM,TXT,RAW] = xlsread('train.xlsx');

filenum = length(TXT) - 1;
LUT = zeros(filenum-1, 3);  % because 7489 doesn't exist

img_side = 192;
final_file = zeros(filenum-1, img_side*img_side+1);

class_index = -1;
process_index = 0;

for k=1:filenum
    
    display(k);
    temp = TXT{k, 1};
    filename = temp;
    templen = length(temp);
    temp = temp(3:templen-4);
    image_index = str2num(temp);
    if image_index==7489
        continue;
    end
    temp = TXT{k, 2};
    templen = length(temp);
    temp = temp(7:templen);
    whale_index = str2num(temp);
    
    process_index = process_index + 1;
    % record in LUT
    LUT(process_index, 1) = image_index;    
    whale_table = LUT(:, 2);
    if sum(find(whale_table==whale_index))==0
        class_index = class_index + 1;
    end
    LUT(process_index, 2) = whale_index;
    LUT(process_index, 3) = class_index;
    
    % resize the image
    rawimg = imread(strcat(dir_folder, '\', filename));
    [rawheight, rawwidth, rawcolor] = size(rawimg);
    if rawcolor==3
        rawimg = rawimg(:,:,3);
    end
    img = imresize(rawimg, [img_side img_side]);
    temp_img = reshape(img, 1, img_side*img_side);
    
    % record in final_file
    final_file(process_index, 1) = class_index;
    final_file(process_index, 2:(img_side*img_side+1)) = temp_img;
    
end

display('Start saving final_file.mat ...');
save final_file.mat final_file LUT
display('Done!');
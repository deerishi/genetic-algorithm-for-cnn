clc;
clear all;
close all;


dir_folder_1 = 'G:\Program Files\Work\deep_learning\head_all';
filelist_1 = ls(dir_folder_1);
filelist_1 = filelist_1(3:length(filelist_1),:);

dir_folder_2 = 'G:\Program Files\Work\deep_learning\imgs';
filelist_2 = ls(dir_folder_2);
filelist_2 = filelist_2(3:length(filelist_2),:);


save_folder = 'G:\Program Files\Work\deep_learning\imgs_head';

parfor i=5001:11468
    if i==7489
        continue;
    end
    down_sam_img = imread(strcat(dir_folder_1, '\w_', num2str(i), '.jpg'));
    original_img = imread(strcat(dir_folder_2, '\w_', num2str(i), '.jpg'));
    croped_im = whale_face_crop(down_sam_img,original_img  );
    imwrite(croped_im, strcat(save_folder, '\w_', num2str(i), '.jpg'), 'jpg');
%     
%     if ((i/100)==round((i/100)))
%         i
%     end
%figure(3), imshow(original_img );
%figure(4), imshow(croped_im)

end

%sublist = 500*rand(nn, 1);
%sublist = floor(sublist);











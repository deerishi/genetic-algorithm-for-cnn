function [ croped_im ] = whale_face_crop( down_sam_img,original_img  )
%WHALE_FACE_CROP Summary of this function goes here
%   Detailed explanation goes here
R = double(down_sam_img(:,:,1));
G = double(down_sam_img(:,:,2));
B = double(down_sam_img(:,:,3));

DS_size=size(R);

[rr,cr]=find(R==255);
[rg,cg]=find(G==0);
[rb,cb]=find(B==0);

Rec=im2bw(R-G-B,0.9);
%figure(1),imshow(Rec)

[height, width, iscolor] = size(Rec);
if width>25
    img = imresize(Rec, 1/0.1);
else
    img = imresize(Rec, 1/0.2);
end

%figure(2),imshow(img)
[r_Rec,c_Rec]=find(img==1);
max_x=max(c_Rec);
min_x=min(c_Rec);
max_y=max(r_Rec);
min_y=min(r_Rec);
width=abs((max_x)-(min_x));
height=abs((max_y)-(min_y));

[newheight, newwidth, iscolor] = size(img);

tlx = max(round(min_x-width/4),1);
tly = max(round(min_y-height/4),1);
rbx = min(round(min_x+width/4*5),newwidth);
rby = min(round(min_y+height/4*5),newheight);

rect=[tlx tly (rbx-tlx) (rby-tly)];



croped_im = imcrop(original_img,rect);

end


function outputimg = detect_whale_face(rawimg)

%---------------------------------------------------------
% Several points:
% 1 - The sea is always blue, sometime dark blue, or even close to black
% 2 - Most part of image is sea, its color has consistency.
% 3 - The color of the patterns on whale's head are lighter than sea, close
% to white
%---------------------------------------------------------

%clc;  clear all;  close all;  rawimg = imread('G:\Program Files\Work\deep_learning\imgs\w_2145.jpg');  %
[height, width, iscolor] = size(rawimg);
if width>250
    img = imresize(rawimg, 0.1);
else
    img = imresize(rawimg, 0.2);
end
r = double(img(:,:,1));
g = double(img(:,:,2));
b = double(img(:,:,3));
hsv = rgb2hsv(img);
hue = hsv(:,:,1);
colorsum = ceil(b+g+r);
copy_colorsum = colorsum;
colordist = abs(r-g) + abs(r-b) + abs(g-b);
copy_colordist = colordist;

%% part1

b = b/max(max(b))*255;
r = r/max(max(b))*255;
[height, width] = size(b);
allmask = zeros(height, width);

temp = ceil(b);
[h, sumh] = generate_hist(temp, allmask, 255);

% this threshold is used to get rid of 85% area where has low intensity in blue channel
temp = sumh(255)*0.9;
for i=255:-1:1
    if sumh(i)<=temp
        th = i;
        break;
    end
end

bw = zeros(height, width);
bw(b>th) = 1;
bw = select_largest_region(bw, 10);

% bw contains regions of white water waves and white whale patterns

%% part2

% use colordist to detect whale
colordist(bw==1) = 384;  %get rid of white water waves and white whale patterns
part2 = ceil(colordist);
[h2, sumh2] = generate_hist(part2, bw, 384);

% this threshold is used to select 5% area where low in colordist
ave_dist = sum(sum(copy_colordist))/height/width;
if ave_dist>20
    temp = sumh2(384)*0.1;
    for i=1:384
        if sumh2(i)>=temp
            th2 = i;
            break;
        end
    end

    bw2 = zeros(height, width);
    bw2(part2<=th2) = 1;
else
    % when the water has low color_dist, the pattern will has high, for some
    % reason
    bw2 = zeros(height, width);
    bw2(part2>40) = 1;
    bw2(part2==384) = 0;
end

%% part3
% use colorsum to detect whale
colorsum(bw==1) = 768;
part3 = ceil(colorsum);
[h3, sumh3] = generate_hist(part3, bw, 768);

% this threshold is used to select 5% area where low in colorsum
temp = sumh3(768)*0.05;
for i=1:768
    if sumh3(i)>=temp
        th3 = i;
        break;
    end
end

bw3 = zeros(height, width);
bw3(part3<=th3) = 1;


%% part4
% combine mask bw, bw2, bw3

whalemask = zeros(height, width);
whalemask(bw>0) = 1;
whalemask(bw2>0) = 1;
whalemask(bw3>0) = 1;

% select 10 largest regions
newwhalemask = select_largest_region(whalemask, 10);
whalemask = imfill(newwhalemask);

part4 = ceil(copy_colordist);

ave_dist = sum(sum(part4))/height/width;
% if the color of water have large color_dist
if ave_dist>20
    
    part4(part4==0) = 1;
    newpart4 = part4;
    
    %{
    [h4, sumh4] = generate_hist(part4, whalemask, 384);
    sumh4 = sumh4/sumh4(384)*255;
    for i=1:height
        for j=1:width
            if whalemask(i,j)==1
                newpart4(i,j) = sumh4(part4(i,j));
            end
        end
    end
    %}

    bw4 = zeros(height, width);
    for i=1:height
        for j=1:width
            if whalemask(i,j)==1 && (newpart4(i,j)<(ave_dist-10))
                bw4(i,j) = 1;
            end
        end
    end
    bw4 = select_largest_region(bw4, 10);
    bw4 = imfill(bw4);
    whalemask = bw4;
end

%% part5

part5 = ceil(copy_colorsum/3);
ave_sum = sum(sum(part5))/height/width;

if ave_sum<200 && ~exist('bw4', 'var')
    part5(part5==0) = 1;
    newpart5 = part5;
    
    %%{
    [h5, sumh5] = generate_hist(part5, whalemask, 255);
    sumh5 = sumh5/sumh5(255)*255;
    for i=1:height
        for j=1:width
            if whalemask(i,j)==1
                newpart5(i,j) = sumh5(part5(i,j));
            end
        end
    end

    masklabel = bwlabel(whalemask, 8);
    labelnum = max(max(masklabel));
    labelarea = zeros(labelnum, 1);
    labelinst = zeros(labelnum, 1);
    for i=1:height
        for j=1:width
            if masklabel(i,j)>0
                labelinst(masklabel(i,j)) = labelinst(masklabel(i,j))+newpart5(i,j);
                labelarea(masklabel(i,j)) = labelarea(masklabel(i,j))+1;
            end
        end
    end
    labelinst = labelinst./labelarea;
    %}
    
    discardindex = find(labelinst<=20);
    temp = masklabel;
    for i=1:length(discardindex)
        temp(masklabel==discardindex(i)) = 0;
    end
    bw5 = zeros(height, width);
    bw5(temp>0) = 1;
    whalemask = bw5;
end

%% part6

SE1=strel('square', 3);
whalemask=imdilate(whalemask, SE1);
masklabel = bwlabel(whalemask, 8);
whalemask=imerode(whalemask, SE1);
masklabel = masklabel.*whalemask;

s = regionprops(masklabel, 'Area');
maxindex = max(max(masklabel));
totalpoints = zeros(maxindex, 1);

% give large regions points (region must contain the whale body)
if exist('bw4', 'var')
    rgarea = zeros(maxindex, 1);
    for i=1:maxindex
        rgarea(i) = s(i).Area;
    end
    [zz, nindex] = sort(rgarea);
    nindex = fliplr(nindex.');
    for i=1:min(maxindex,4)
        totalpoints(nindex(i)) = totalpoints(nindex(i)) + (5-i);
    end
end

% give regions with 0.9 eccentricity points
s = regionprops(masklabel, 'Eccentricity');
rgecc = zeros(maxindex, 1);
for i=1:maxindex
    rgecc(i) = abs(0.925 - s(i).Eccentricity);
end
[zz, nindex] = sort(rgecc);
for i=1:min(maxindex,4)
    %totalpoints(nindex(i)) = totalpoints(nindex(i)) + (5-i);
end

% give regions with large area of low colordist points
if exist('bw4', 'var')
    temp4 = newpart4.*whalemask;
    temp4(temp4>(ave_dist/2)) = 0;
    temp4(temp4>0) = 1;
    temp4(r>(ave_sum+50)) = 0;
    temp4 = masklabel.*temp4;
    temparea = zeros(maxindex, 1);
    for i=1:height
        for j=1:width
            if temp4(i,j)>0
                temparea(temp4(i,j)) = temparea(temp4(i,j)) + 1;
            end
        end
    end
    
    %temparea = temparea./rgarea;
    [zz, nindex] = sort(temparea);
    nindex = fliplr(nindex.');
    for i=1:min(maxindex,4)
        totalpoints(nindex(i)) = totalpoints(nindex(i)) + 2*(5-i);
    end
end

% give large regions with high red channel points
temp5 = r.*whalemask;
temp5(temp5<(ave_sum-30)) = 239;
temp5(temp5<(ave_sum+50)) = 0;
temp5(temp5>240) = 0;
temp5(temp5>0) = 1;
temp5 = masklabel.*temp5;
temparea = zeros(maxindex, 1);
for i=1:height
    for j=1:width
        if temp5(i,j)>0
            temparea(temp5(i,j)) = temparea(temp5(i,j)) + 1;
        end
    end
end
[zz, nindex] = sort(temparea);
nindex = fliplr(nindex.');
for i=1:min(maxindex,4)
    totalpoints(nindex(i)) = totalpoints(nindex(i)) + (5-i);
end

% if color_dist is not used, double use red channel
if ~exist('bw4', 'var')
    for i=1:min(maxindex,4)
        totalpoints(nindex(i)) = totalpoints(nindex(i)) + (5-i);
    end
end

%%
toppoint = max(totalpoints);
toppointindex = find(totalpoints==toppoint);
whalemask = zeros(height, width);
for i=1:length(toppointindex)
    whalemask(masklabel==toppointindex(i)) = 1;
end

s = regionprops(whalemask, 'Area');
maskarea = s.Area/height/width;

copy_whalemask = whalemask;

if maskarea>0.02 && exist('bw4', 'var') && (ave_dist>40)
    water = bw4 - bw2;
    %whalemask(water>0) = 0;  % remove some waves and whale patterns, keep the whale body
    whalemask(part5>(ave_sum+50)) = 0;
    if ave_dist>60
        whalemask(copy_colordist>(ave_dist/2)) = 0;
    end
    whalemask = select_largest_region(whalemask, 10);
    copy_whalemask = whalemask;
    
    s = regionprops(whalemask, 'BoundingBox');
    ulcenter = ceil(s.BoundingBox(1:2));
    s = regionprops(whalemask, 'ConvexImage');
    tempmask = s.ConvexImage;
    [temph, tempw] = size(tempmask);
    for i=1:temph
        for j=1:tempw
            if tempmask(i,j)==1
                whalemask(i+ulcenter(2), j+ulcenter(1)) = 1;
            end
        end
    end
end

s = regionprops(whalemask, 'Centroid');
center = s.Centroid;
s = regionprops(whalemask, 'Orientation');
angle = s.Orientation/180*pi;
s = regionprops(whalemask, 'MajorAxisLength');
axislength = s.MajorAxisLength/2;

headx1 = round(center(1)+ axislength*0.9*cos(angle));
headx2 = round(center(1)- axislength*0.9*cos(angle));
heady1 = round(center(2)- axislength*0.9*sin(angle));
heady2 = round(center(2)+ axislength*0.9*sin(angle));

dist1 = (headx1-width/2)^2 + (heady1-height/2)^2;
dist2 = (headx2-width/2)^2 + (heady2-height/2)^2;
dist1 = sqrt(dist1);
dist2 = sqrt(dist2);

outputimg = img;

%{
bound = bwperim(whalemask);
for i=1:height
    for j=1:width
        if whalemask(i,j)==1
            %outputimg(i,j,1) = 255;
        end
        if bound(i,j)==1
            outputimg(i,j,1) = 255;
            outputimg(i,j,2) = 255;
            outputimg(i,j,3) = 0;
        end
    end
end
%}

boxhfedge = round(axislength*0.4);
boxhfedge = max(boxhfedge, round(width/10));

% case1
headx = headx1;
heady = heady1;
upleftx = max(1, (headx-boxhfedge));
uplefty = max(1, (heady-boxhfedge));
btrightx = min(width, (headx+boxhfedge));
btrighty = min(height, (heady+boxhfedge));
headshape1 = copy_whalemask(uplefty:btrighty, upleftx:btrightx);
[temph, tempw] = size(headshape1);

headimg1 = img(uplefty:btrighty, upleftx:btrightx, :);
headr = double(headimg1(:, :, 1));
headmask1 = zeros(temph, tempw);
headmask1(headr>(ave_sum+50)) = 1;
headr1 = headmask1;
%headmask1 = select_largest_region(headmask1, 1);
headmask1 = double(headmask1|headshape1);
headmask1 = imfill(headmask1);
s = regionprops(headmask1, 'Area');

if max(max(headmask1))>0
    headarea1 = s.Area/temph/tempw;
    headarea1 = abs(headarea1-3/16)/3*16;
    
    % convert contour from descartes to polar
    s = regionprops(headmask1, 'Centroid');
    headcenter1 = s.Centroid;
    bound1 = bwperim(headmask1);
    s = regionprops(bound1, 'PixelList');
    boundlist1 = s(1).PixelList;
    if length(s)>1
        for i=2:length(s)
            boundlist1 = vertcat(boundlist1, s(i).PixelList);
        end
    end
    boundlist1(:,1) = boundlist1(:,1) - headcenter1(1);
    boundlist1(:,2) = boundlist1(:,2) - headcenter1(2);
    [t1, r1] = cart2pol(boundlist1(:,1), boundlist1(:,2));
    [t1, index1] = sort(t1);
    temp = r1;
    for i=1:length(temp)
        r1(i) = temp(index1(i));
    end
    r11 = smooth(r1, 'moving');
    r11 = smooth(r11, 'moving');
    r11 = smooth(r11, 'moving');
    r11 = smooth(r11, 'moving');
    diffr1 = abs(r1-r11);
    ratio1 = sum(diffr1)/length(temp);
    
else
    headarea1 = 6;
    ratio1 = 99;
end



%case2
headx = headx2;
heady = heady2;
upleftx = max(1, (headx-boxhfedge));
uplefty = max(1, (heady-boxhfedge));
btrightx = min(width, (headx+boxhfedge));
btrighty = min(height, (heady+boxhfedge));
headshape2 = copy_whalemask(uplefty:btrighty, upleftx:btrightx);
[temph, tempw] = size(headshape2);
headshape2 = imfill(headshape2);

headimg2 = img(uplefty:btrighty, upleftx:btrightx, :);
headr = double(headimg2(:, :, 1));
headmask2 = zeros(temph, tempw);
headmask2(headr>(ave_sum+50)) = 1;
headr2 = headmask2;
%headmask2 = select_largest_region(headmask2, 1);
headmask2 = double(headmask2|headshape2);
headmask2 = imfill(headmask2);
s = regionprops(headmask2, 'Area');
if max(max(headmask2))>0
    headarea2 = s.Area/temph/tempw;
    headarea2 = abs(headarea2-3/16)/3*16;
    % convert contour from descartes to polar
    s = regionprops(headmask2, 'Centroid');
    headcenter2 = s.Centroid;
    bound2 = bwperim(headmask2);
    s = regionprops(bound2, 'PixelList');
    boundlist2 = s(1).PixelList;
    if length(s)>1
        for i=2:length(s)
            boundlist2 = vertcat(boundlist2, s(i).PixelList);
        end
    end
    boundlist2(:,1) = boundlist2(:,1) - headcenter2(1);
    boundlist2(:,2) = boundlist2(:,2) - headcenter2(2);
    [t2, r2] = cart2pol(boundlist2(:,1), boundlist2(:,2));
    [t2, index2] = sort(t2);
    temp = r2;
    for i=1:length(temp)
        r2(i) = temp(index2(i));
    end
    r22 = smooth(r2, 'moving');
    r22 = smooth(r22, 'moving');
    r22 = smooth(r22, 'moving');
    r22 = smooth(r22, 'moving');
    diffr2 = abs(r2-r22);
    ratio2 = sum(diffr2)/length(temp);
    
else
    headarea2 = 6;
    ratio2 = 99;
end



if dist1<dist2
    headx = headx1;
    heady = heady1;
    if (ratio1>(3*ratio2)) && (dist1>(dist2/2))
        headx = headx2;
        heady = heady2;
    end
    if length(find(headr1==1))<20
        headx = headx2;
        heady = heady2;
    end
    if headarea1>2 && headarea2<0.5
        headx = headx2;
        heady = heady2;
    end
end

if dist2<dist1
    headx = headx2;
    heady = heady2;
    if (ratio2>(3*ratio1)) && (dist2>(dist1/2))
        headx = headx1;
        heady = heady1;
    end
    if length(find(headr2==1))<20
        headx = headx1;
        heady = heady1;
    end
    if headarea2>2 && headarea1<0.5
        headx = headx1;
        heady = heady1;
    end
end


upleftx = max(1, (headx-boxhfedge));
uplefty = max(1, (heady-boxhfedge));
btrightx = min(width, (headx+boxhfedge));
btrighty = min(height, (heady+boxhfedge));

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
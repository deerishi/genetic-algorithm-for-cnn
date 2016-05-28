
clc;
close all;
clear all;

load final_file.mat;

[height, width] = size(final_file);

class = LUT(:,3);
classnum = zeros(max(class)+1, 1);
for i = 1 : max(class)+1
    classnum(i) = length(find(class==(i-1)));
end
testnum = ceil(classnum*0.15);
trainnum = classnum - testnum;
temp = length(find(trainnum==0));

LUT1 = zeros(sum(trainnum)+temp, 3);
file1 = zeros(sum(trainnum)+temp, width);
LUT2 = zeros(sum(testnum), 3);
file2 = zeros(sum(testnum), width);

k1 = 1;  k2 = 1;  k = 1;
for nn = 0 : max(class)
    index = find(class==nn);
    if length(index)==1
        LUT1(k1,:) = LUT(k,:);
        LUT2(k2,:) = LUT(k,:);
        file1(k1,:) = final_file(k,:);
        file2(k2,:) = final_file(k,:);
        k1 = k1+1;
        k2 = k2+1;
        k = k+1;
        continue;
    end
    for i=1:trainnum(nn+1)
        LUT1(k1,:) = LUT(k,:);
        file1(k1,:) = final_file(k,:);
        k1 = k1+1;
        k = k+1;
    end
    for i=1:testnum(nn+1)
        LUT2(k2,:) = LUT(k,:);
        file2(k2,:) = final_file(k,:);
        k2 = k2+1;
        k = k+1;
    end
end

%%{

filename = 'train_lut.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'image_index,whale_index,whale_class\n');
fclose(fid);
dlmwrite(filename, LUT1, '-append', 'delimiter', ',');

filename = 'test_lut.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'image_index,whale_index,whale_class\n');
fclose(fid);
dlmwrite(filename, LUT2, '-append', 'delimiter', ',');

filename = 'train_data.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'whale_class');
for i=1:192*192
    fprintf(fid, strcat(',data', num2str(i)));
end
fprintf(fid, '\n');
fclose(fid);
dlmwrite(filename, file1, '-append', 'delimiter', ',');

filename = 'test_data.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'whale_class');
for i=1:192*192
    fprintf(fid, strcat(',data', num2str(i)));
end
fprintf(fid, '\n');
fclose(fid);
dlmwrite(filename, file2, '-append', 'delimiter', ',');

%}
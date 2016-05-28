function [h, sumh] = generate_hist(img, mask, max_intensity)

[height, width] = size(img);
img(img==0) = 1;
sumh = zeros(max_intensity, 1);
h = zeros(max_intensity, 1);

for i=1:height
    for j=1:width
        if mask(i,j)<1
            h(img(i,j)) = h(img(i,j))+1;
        end
    end
end

sumh(1) = h(1);
for i=2:max_intensity
    sumh(i) = h(i)+sumh(i-1);
end
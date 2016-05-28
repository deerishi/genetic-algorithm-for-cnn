function newmask = select_largest_region(mask, nn)

[height, width] = size(mask);
newmask = zeros(height, width);

templabel = bwlabel(mask, 4);
maxindex = max(max(templabel));
rgarea = zeros(maxindex, 1);
if maxindex>nn
    s = regionprops(templabel, 'Area');
    for i=1:maxindex
        rgarea(i) = s(i).Area;
    end
    [zz, nindex] = sort(rgarea);
    nindex = fliplr(nindex.');
    for i=1:nn
        newmask(templabel==nindex(i)) = 1;
    end
else
    newmask = mask;
end
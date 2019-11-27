function [] = showlabel(IM,L)
%IM and L are 2D matrices. 

IM=double(IM);
figure,imshow(IM,[]);hold on
B = bwboundaries(L>0);
CC=bwconncomp(L>0);
xcentroid=nan(numel(CC.PixelIdxList),1);
ycentroid=nan(numel(CC.PixelIdxList),1);
for i=1:numel(CC.PixelIdxList)
    [xx,yy]=ind2sub(size(IM),CC.PixelIdxList{i});
    xcentroid(i)=sum(IM(CC.PixelIdxList{i}).*xx)./sum(IM(CC.PixelIdxList{i}));
    ycentroid(i)=sum(IM(CC.PixelIdxList{i}).*yy)./sum(IM(CC.PixelIdxList{i}));
    1;
end
visboundaries(B)
1;
%plot(ycentroid,xcentroid,'og','LineWidth',3);hold on
drawnow;
end


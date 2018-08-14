%This is to check if a topologically aware fast marching code provides a
%better cell boundary for automated ground truth generation from cell
%centerpoints. 

%{
%Dataset 1
IM = imread('/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/53_raw.tif');
L = imread('/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v4/53_labels.tif');
L2 = imread('/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v3/53_labels.tif');
IM = IM(400:700,1400:1700,1);
L = L(400:700,1400:1700,1);

IM = IM(60:120,30:90);
L = L(60:120,30:90);

%Dataset 2

IM = imread('/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/119_raw.tif');
L = imread('/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v4/119_labels.tif');
IM = IM(560:1090,790:1320);
L = L(560:1090,790:1320);
%}
function [S,D,T] = TestTopoFM(IM,L)
IM = double(IM);
pad = 2;%Matches the fast marching padding
C = bwconncomp(L==1);
X = nan(numel(C.PixelIdxList),1);
Y = nan(numel(C.PixelIdxList),1);
for i = 1:numel(C.PixelIdxList)
    [xcell,ycell] = ind2sub(size(L),C.PixelIdxList{i});
    X(i) = mean(xcell);
    Y(i) = mean(ycell);
end
%figure(100);imshow(padarray(IM,[pad,pad,0]),[]);hold on;
%plot(Y+pad,X+pad,'xr','LineWidth',2);hold on
SVr = [X,Y,ones(size(X))];
[S,~,D,T]=FastMarchingTube((IM./max(IM(:))).^2,SVr,40,[1,1,1]);
%figure,imshow(IM,[]);hold on;c = contour(T,[9,15,20,30]);

%figure,imshow(IM,[]);hold on;
%B1 = bwboundaries(T<=25);
%B2 = bwboundaries(L==1);
%visboundaries(gca,B1);plot(S(:,2),S(:,1),'.r','MarkerSize',5)
%visboundaries(gca,B2,'Color','c');
end

function [Labels,Tadapt] = gen_Tadapt(IM,S,D,T)
padsize = 20;
typicalcellsize=10;
small_region_thr = 10;

padsizevec = [padsize,padsize];

[cell_x,cell_y] = find(T==0);
cell_r = [cell_x,cell_y];

IM = padarray(IM,padsizevec);
T = padarray(T,padsizevec,inf);
D = padarray(D,padsizevec,max(D(:)));
S = S + padsize;
cell_r = cell_r + padsize;

S_ind = sub2ind(size(D),S(:,1),S(:,2));
D(S_ind)=max(D(:));

%Ignore beyond the typical cell radius
D(D>typicalcellsize)=typicalcellsize+1;
T(D>typicalcellsize)=inf;
L = bwlabel(D<=typicalcellsize,4);
unique_lbl = unique(L(L>0)); %0 is background

%Remove small regions (regions FastMarching did not propagate into)
for l = 1:numel(unique_lbl)
    if sum(L==unique_lbl(l))<small_region_thr
        L(L==unique_lbl(l))=0;
    end
end

timer=tic();
Tadapt = inf(size(T));
nhood = padsize;
for i = 1:numel(cell_r(:,1))
    cell_r_int = round(cell_r(i,:));
    xrange=cell_r_int(1)-nhood:cell_r_int(1)+nhood;
    yrange=cell_r_int(2)-nhood:cell_r_int(2)+nhood;
    Tcut = T(xrange,yrange);
    Lcut = L(xrange,yrange);
    
    Tcut(Lcut~=L(cell_r_int(:,1),cell_r_int(:,2))) = inf;
    try
        Tcut = Tcut./median(Tcut(~isinf(Tcut)));
    catch
        disp('Failed to find label');
        Tcut = inf(size(Tcut));
    end
    Tadapt(xrange,yrange) = min(Tcut,Tadapt(xrange,yrange));
end
toc(timer)

mask = false(size(T));
mask(S_ind)=true;
mask = imdilate(mask,strel('disk',1));
Tadapt(mask)=inf;

%Heuristic morphological operations to obtain reasonable labels
foreground_mask = Tadapt<=0.8;
boundary_mask = Tadapt<=1.5;
boundary_mask = imdilate(boundary_mask ,strel('disk',1));
boundary_mask = imdilate(boundary_mask ,strel('disk',1));
boundary_mask = imdilate(boundary_mask ,strel('disk',1));
boundary_mask = imerode(boundary_mask ,strel('disk',1));
boundary_mask = imerode(boundary_mask ,strel('disk',1));

%{
figure,imshow(IM,[]);hold on;
B1 = bwboundaries(foreground_mask);
B2 = bwboundaries(boundary_mask);
visboundaries(gca,B1,'Color','b');
visboundaries(gca,B2,'Color','r');
%}

Labels = uint8(zeros(size(foreground_mask)));
Labels(boundary_mask) = 1;
Labels(foreground_mask) = 2;
Labels = Labels(padsize+1:end-padsize,padsize+1:end-padsize);
Tadapt = Tadapt(padsize+1:end-padsize,padsize+1:end-padsize);
end
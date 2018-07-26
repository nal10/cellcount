%load file into matrix IM


base_path = '/home/rohan/';
raw_dir = '/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/';
proc_dir = '/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v3/';
save_dir = '/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v4/';

%Orig = imread('/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/53_raw.tif');
%IM = imread('/home/rohan/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v3/53_labels.tif');

fname = dir([base_path,raw_dir]);
fname = {fname.name};
fname = {fname{contains(fname,'.tif')}}';
fname_label = strrep(fname,'raw','labels');

for i = 1:numel(fname)
    
    Orig = imread([base_path,raw_dir,fname{i}]);
    IM = imread([base_path,proc_dir,fname_label{i}]);
    
    dthr = 1.5;
    bw = logical(IM);
    L = bwdist(1-bw);
    L(L>0 & L<dthr) = 1;
    L(L>=dthr) = 2;
    
    r=3;
    n=8;
    SE = strel('disk',r,n);
    bwdil = double(imdilate(bw,SE))*2;
    
    conv_L = double(bwconvhull(L==2,'objects'))*3;
    M = max(bwdil,conv_L);
    
    % figure;imagesc(M);axis equal;hold on
    % title('Convex foreground')
    % B1 = bwboundaries(L>=1);
    % visboundaries(B1)
    % B2 = bwboundaries(M==3);
    % visboundaries(B2)
    % drawnow;
    
    figure;imshow(Orig);axis equal;hold on
    title('original overlay')
    ax = gca;
    B1 = bwboundaries(M>=1);
    visboundaries(ax,B1);
    
    B2 = bwboundaries(M==3);
    visboundaries(ax,B2,'Color','b')
    drawnow;
    
    imwrite(M,[base_path,save_dir,fname_label{i}])
end
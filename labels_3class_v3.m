function [] = labels_3class_v3(f1,f2)
%This version is used with version 2 annotations.
im_dir = '/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
csv_dir = '/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
mat_dir = '/home/rohan/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_02_Mat_v1/';

csvfile = dir([im_dir]);
csvfile = {csvfile.name};
csvfile = {csvfile{contains(csvfile,'-optim.csv')}}';
IM_file = strrep(csvfile,'-optim.csv','.tif');

%Parameters: 
FM_max_dist = 40; 
gauss_sigma = 2;

%for i = 1:numel(csvfile)    
for i = f1:f2
    IM = tifvol2mat([im_dir,IM_file{i}],[],[]);IM=IM(:,:,1);
    r = csvread([csv_dir,strrep(csvfile{i},'.tif','.csv')]);
    SVr = [r(:,2),r(:,1),ones(size(r,1))];
    imfilt = imgaussfilt(IM./max(IM(:)),gauss_sigma);
    [S,~,D,T]=FastMarchingTube(imfilt,SVr,FM_max_dist,[1,1,1]);
    
    %{
    figure,imshow(IM,[]);hold on;
    c = contour(T,[10,20,25,30]);
    plot(S(:,2),S(:,1),'.r','MarkerSize',10)
    B1 = bwboundaries(T<=500);
    visboundaries(gca,B1,'Color','c');
    %}
    save([mat_dir,strrep(csvfile{i},'-optim.csv','.mat')],'S','D','T','IM')
end
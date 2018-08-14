%load file into matrix IM

if ismac
    base_path = '/Users/fruity';
else
    base_path = '/home/rohan/';
end
raw_dir = '/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/';
proc_dir = '/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v3/';
save_dir = '/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v5/';

%Orig = imread('/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_01_Images/53_raw.tif');
%IM = imread('/home/rohan/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_01_Labels_v3/53_labels.tif');

fname = dir([base_path,raw_dir]);
fname = {fname.name};
fname = {fname{contains(fname,'.tif')}}';
fname_label = strrep(fname,'raw','labels');
fname_mat = strrep(fname,'raw.tif','temp.mat');
for i = 1:numel(fname)
    load([base_path,save_dir,fname_mat{i}],'S','D','T','IM','L');
    Sind = sub2ind(size(IM),S(:,1),S(:,2));    

    simple_px=false(size(IM));
    simple_px(Sind)=true;
    simple_px(D>20 | T>50)=0; %Ignore far away non simple points 
    
    
    boundary_px = imdilate((T>20 & T<40),strel('disk',1,8));
    foreground_px = T<=20;
    
    simple_px(~foreground_px & ~boundary_px)=false;
    simple_px = imdilate(simple_px, strel('disk',2,8));
    
    
    M = uint8(zeros(size(T)));
    M(boundary_px)=2;
    M(foreground_px)=1;
    M((M==1) & ~imerode(M==1,strel('disk',1,8)))=2;
    M(simple_px)=2;
    
    %{
    figure,imshow(IM,[])
    B1 = bwboundaries(M==1);
    B2 = bwboundaries(M==2);
    visboundaries(gca,B1,'color',[1 0 0])
    visboundaries(gca,B2,'color',[0 0.3 0.8])
    drawnow
    figure,imshow(M,[])
    %}
    
    imwrite(uint8(M),[base_path,save_dir,fname_label{i}])
    %save([base_path,save_dir,fname_mat{i}],'S','D','T','IM','L')
end
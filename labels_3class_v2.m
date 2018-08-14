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
    
    IM = imread([base_path,raw_dir,fname{i}]);IM=IM(:,:,1);
    L = imread([base_path,proc_dir,fname_label{i}]);
    [S,D,T] = TestTopoFM(IM,L);
    
    save([base_path,save_dir,fname_mat{i}],'S','D','T','IM','L')
end
mat_dir = '/home/rohan/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_02_Mat_v1/';
labels_dir = '/home/rohan/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_02_Labels_v1/';
fnames = dir([mat_dir,'*.mat']);fnames = {fnames.name}';
for f=1:numel(fnames)
    F = load([mat_dir,fnames{f}]);
    [Labels,~] = gen_Tadapt(F.IM,F.S,F.D,F.T);
    labelfile = strrep([labels_dir,fnames{f}],'.mat','_labels.tif');
    imwrite(Labels,labelfile)
end



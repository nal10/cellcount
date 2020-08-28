mat_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Unet_tiles_082020/';
labels_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Unet_tiles_082020/';
fnames = dir([mat_dir,'*.mat']);fnames = {fnames.name}';
for f=1:numel(fnames)
    F = load([mat_dir,fnames{f}]);
    [Labels,~] = gen_lbl_from_mat(F.IM,F.S,F.D,F.T);
    labelfile = strrep([labels_dir,fnames{f}],'.mat','_labels.tif');
    imwrite(Labels,labelfile)
end
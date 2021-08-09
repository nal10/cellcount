mat_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/control_retraining/';
labels_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/control_retraining/';
fnames = dir([mat_dir,'*.mat']);fnames = {fnames.name}';
for f=1:numel(fnames)
    F = load([mat_dir,fnames{f}]);
    [Labels,~] = lbl_from_FM(F.IM,F.S,F.D,F.T);
    labelfile = strrep([labels_dir,fnames{f}],'.mat','_labels.tif');
    imwrite(Labels,labelfile)
    drawnow;
end
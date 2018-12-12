%function [X] = loaddataset_v2()
%dat_path is directory of all images
%typ is 'label' or 'ensembleLabels' or 'raw' or 'EnsembleLabels'
%output is a 3d stack
%assumes all files have the same dimensions


im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
csv_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';

fname=dir(im_path);
fname={fname(:).name}';
IM_files=fname(contains(fname,'tif'));

fname=dir(csv_path);
fname={fname(:).name}';
csv_files=fname(contains(fname,'csv'));


for i=1:numel(IM_files)
    try
        if strcmp(IM_files{1}(1:end-4), csv_files{1}(1:end-4))
            IM=tifvol2mat([im_path,IM_files{i}],[],[]);
            r = csvread([csv_path,csv_files{i}]);
            success=1;
        else
            success=0;
        end
    catch
        success=0;
        
    end
    if success
        
        figure,
        imshow(IM,[]);hold on;plot(r(:,1),r(:,2),'or')
        title(IM_files{i})
        drawnow;
        
        %Perform optimization
        [final_pos] = optim_v4(IM,r);
        title(IM_files{i})
        drawnow;
        IM = [];
        r = [];
        
    else
        disp(i)
    end
end

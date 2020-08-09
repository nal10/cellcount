function [X] = optim_centers_script()
%This function optimizes manual markings of cell body centers.


im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/';
csv_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/';

fname=dir(im_path);
fname={fname(:).name}';
IM_files=fname(contains(fname,'red.tif'));

for i=1:numel(IM_files)
    try
        IM = [];
        r = [];
        IM = tifvol2mat([im_path,IM_files{i}],[],[]);
        %r = csvread([csv_path,strrep(IM_files{i},'.tif','.csv')]);
        r = readmatrix([csv_path,strrep(IM_files{i},'.tif','.csv')]);
        r = r(:,2:3);
        success=true;
    catch
        disp(['Failed to load ',IM_files{i}])
        success=false;
    end
    if success
        %Perform optimization
        [rnew] = optim_centers(IM,r,1);
        drawnow;
        
        %Check image
        figure,
        imshow(IM,[]);hold on;
        plot(r(:,1),r(:,2),'or')
        plot(rnew(:,1),rnew(:,2),'ob')
        caxis([0 30])
        title(IM_files{i})
        drawnow;
        
        %Write to new csv
        csvwrite([csv_path,strrep(IM_files{i},'.tif','-optim.csv')],rnew);
     
    end
end

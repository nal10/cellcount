function [X] = gen_optimized_annotation()
%This function optimizes manual markings of cell body centers.


im_path='/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
csv_path='/home/rohan/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';

fname=dir(im_path);
fname={fname(:).name}';
IM_files=fname(contains(fname,'.tif'));

for i=1:numel(IM_files)
    try
        IM = [];
        r = [];
        IM = tifvol2mat([im_path,IM_files{i}],[],[]);
        r = csvread([csv_path,strrep(IM_files{i},'.tif','.csv')]);
        success=true;
    catch
        disp(['Failed to load ',IM_files{i}])
        success=false;
    end
    if success
        %Perform optimization
        [rnew] = optim_v4(IM,r,0);
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

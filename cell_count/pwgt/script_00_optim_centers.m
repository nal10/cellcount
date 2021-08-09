function [] = optim_centers_script()
%This function optimizes manual markings of cell body centers.


%im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/';
%csv_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Unet_tiles_082020/';
im_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/control_retraining/';
csv_path='/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/control_retraining/';

fname=dir(im_path);
fname={fname(:).name}';
IM_files=fname(contains(fname,'green.tif'));

for i=1:numel(IM_files)
    try
        IM = [];
        r = [];
        IM = tifvol2mat([im_path,IM_files{i}],[],[]);
        csv_file = [csv_path,strrep(IM_files{i},'.tif','.csv')];
        if isfile(csv_file)
            r = readmatrix(csv_file);
            r = r(:,2:3);
        else
            print(['No annotation found. Assume no cells in:', IM_files{i}])
            r = nan(1,2);
        end
        %Handle empty annotation file:
        if all(isnan(r))
            r = int64.empty(0,2);
        end
        success=true;
    catch
        disp(['Failed to load ',IM_files{i}])
        success=false;
    end
    if success
        %Perform optimization
        if size(r,1)>0
            [~,jj] = rem_duplicates(r,1);
            r(jj,:) = [];
            [rnew] = optim_centers(IM,r,0);
            
            drawnow;
            
            %Check image
            figure,
            imshow(IM,[]);hold on;
            plot(r(:,1),r(:,2),'or','LineWidth',5,'MarkerSize',5)
            plot(rnew(:,1),rnew(:,2),'ob','LineWidth',5,'MarkerSize',5)
            caxis([0 30])
            title(IM_files{i})
            drawnow;
        else
            rnew=r;
        end
        %Write to new csv
        csvwrite([csv_path,strrep(IM_files{i},'.tif','-optim.csv')],rnew);
    end
end

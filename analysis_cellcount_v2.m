function [counts,diffcount] = analysis_cellcount_v2()

train_fileid = {'268778_157','268778_77',...
                '271317_143','271317_95',...
                '275376_147','275376_175',...
                '275705_109','275705_131',...
                '279578_111','279578_123',...
                '299769_119','299769_167',...
                '299773_57','299773_68',...
                '301199_74','315576_116',...
                '316809_60','324471_113',...
                '371808_52','386384_47',...
                '387573_103_1','389052_108',...
                '389052_119'};
            
val_fileid = {'330689_118','371808_60',...
              '387573_103','324471_53'};

%omitted '333241_113' - this has no cells.
          
imlist = val_fileid;
thr = 0.5;
dthr = 10;

cch = [0 0.3 1];
cca = [1 0 0];

count_man = nan(numel(imlist),1);
count_alg = nan(numel(imlist),numel(thr));
for i = 1:numel(imlist)
    seg = load(['/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/results/Data_v2_Run_v1epochs_1500_Segmentations/seg',imlist{i},'.mat']);
    lbl = load(['/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_02_Mat_v1/',imlist{i},'.mat'],'T');
    
    [x,y] = find(lbl.T==0);
    figure,imshow(seg.im,[0,0.2]);hold on
    plot(y,x,'+','LineWidth',2,'Color',cch);
    hold on;B2 = bwboundaries(seg.fg>0.5);visboundaries(gca,B2,'color',[1 0. 0]);
    title(imlist(i));
    drawnow;
    %}
    
    %Data for difference threshold choices:
    for t = 1:numel(thr)
        CC = bwconncomp((seg.fg-seg.bo)>thr(t));
        count_man(i) = numel(x);
        count_alg(i,t) = CC.NumObjects;
    end
    
    CC = bwconncomp((seg.fg-seg.bo)>0.5);
    Xx = nan(CC.NumObjects,1);
    Yy = nan(CC.NumObjects,1);
    for c = 1:numel(CC.PixelIdxList)
        [xcell,ycell] = ind2sub(size(seg.fg),CC.PixelIdxList{c});
        Xx(c) = mean(xcell);
        Yy(c) = mean(ycell);
    end
    
    Dd = ((Xx-x').^2+(Yy-y').^2).^0.5;
    Dd = Dd<dthr;
    
    %{
    figure,imshow(seg.im,[]);hold on
    FP = find(sum(Dd,2)==0); %in Xx and Yy
    FN = find(sum(Dd,1)==0); %in x and y
    plot(y(FN),x(FN),'s','LineWidth',2,'Color',cch)
    plot(Yy(FP),Xx(FP),'o','LineWidth',2,'Color',cca)
    %}
    
    %{
    potFN = find(sum(Dd,2)>1); %in Xx and Yy
    potFP = find(sum(Dd,1)>1); %in x and y
    plot(Yy(potFN),Xx(potFN),'o','LineWidth',2,'Color',cca)
    for ii=1:numel(potFN)
        partnerind = find(Dd(potFN(ii),:));
        plot(y(partnerind),x(partnerind),'s','LineWidth',2,'Color',cch)
    end
    
    
    plot(y(potFP),x(potFP),'s','LineWidth',2,'Color',cch')
    for ii=1:numel(potFP)
        partnerind = find(Dd(:,potFP(ii)));
        plot(Yy(partnerind),Xx(partnerind),'o','LineWidth',2,'Color',cca)
    end
    %}
    
    clear seg lbl
end

    

diffcount = bsxfun(@minus,count_man,count_alg);
disp(diffcount);
counts = [count_man,count_alg];
function [counts,diffcount] = analysis_cellcount_v2()


csv_orig_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
csv_optim_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/raw/Dataset_02_Images/';
seg_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/results/Data_v2_Run_v3epochs_1500_Segmentations/';
mat_dir = '/Users/fruity/Dropbox/AllenInstitute/CellCount/dat/proc/Dataset_02_Mat_v1/';

train_fileid = {'268778_157', '268778_77',...
                '271317_95', '271317_143',...
                '275376_147', '275376_175',...
                '275705_109', '275705_131',...
                '279578_111', '279578_123',...
                '299769_119', '299769_167',...
                '299773_57', '299773_68',...
                '301199_74', '315576_116',...
                '316809_60', '324471_113',...
                '330689_118', '371808_52',...
                '386384_47', '387573_103_1',...
                '389052_108', '389052_119'};
            
val_fileid = {'371808_60', '387573_103',...
              '324471_53'};

%omitted '333241_113' - this has no cells.
          
imlist = {'387573_103'};
%imlist = {train_fileid{:},val_fileid{:}};
%imlist = {'389052_108','371808_52'}
thr = 0.5;
dthr = 5;

cco = [1 0.5 0];
cch = [0 0.3 1];
cca = [1 0 0];
%rorig = csvread([csv_dir,strrep(csvfile{i},'.tif','.csv')]);
%roptim = csvread([csv_dir,strrep(csvfile{i},'.tif','.csv')]);
count_man = nan(numel(imlist),1);
count_alg = nan(numel(imlist),numel(thr));



for i = 1:numel(imlist)
    try
    r_orig = csvread([csv_orig_dir,imlist{i},'.csv']);r_orig = unique(r_orig,'rows');
    r_optim = csvread([csv_optim_dir,imlist{i},'-optim.csv']);
    catch
        r_orig =[];
        r_optim=[];
    end
    seg = load([seg_dir,'seg',imlist{i},'.mat']);
    
    %{
    hf=figure;imshow(seg.im,[0. 0.05]);drawnow;
    fig2png(gcf,400,400,0)
    print(hf, '-dpng', ['/Users/fruity/Desktop/',imlist{i},'.png'])
    close all
    %}
    
    %
    mat = load([mat_dir,imlist{i},'.mat'],'IM','S','D','T');
    [lbl,Tadapt] = gen_Tadapt(mat.IM,mat.S,mat.D,mat.T);
    %Plot annotation before and after optimization-------------------------
    hf = i*100+1;
    figure(hf);clf(hf);movegui(hf,'northwest');
    imshow(seg.im,[0,0.5]);hold on
    plot(r_orig(:,1),r_orig(:,2),'+','LineWidth',2,'Color',cco);
    plot(r_optim(:,1),r_optim(:,2),'+','LineWidth',2,'Color',cch);
    plot([r_orig(:,1),r_optim(:,1)]',[r_orig(:,2),r_optim(:,2)]','-c')
    title(imlist(i),'Interpreter','None');
    drawnow;
    
    
    %Plot Tadapt contour overlay-------------------------------------------
    hf = i*100+2;
    figure(hf);clf(hf);movegui(hf,'north');
    hi = axes('Parent',hf,'Visible','off');axis(hi,'ij');axis(hi,'equal')
    imagesc(seg.im,'Parent',hi);hold on
    colormap(hi,'gray');caxis([0,0.5]);
    plot(r_optim(:,1),r_optim(:,2),'+','LineWidth',2,'Color',cch,'Parent',hi);
    drawnow;
    
    cax = axes('Parent',hf);    
    linkaxes([cax,hi],'xy')
    contour(Tadapt,[0.3:0.3:1.5],'Parent',cax)
    cax.Position = hi.Position;
    axis(cax,'ij');
    cax.Visible='off';
    axis([cax,hi],'square')
    axis([cax,hi],'equal')
    drawnow
    
    
    %Plot Tadapt contour overlay-------------------------------------------
    hf = i*100+3;
    figure(hf);clf(hf);movegui(hf,'northeast');
    hi = axes('Parent',hf,'Visible','off');axis(hi,'ij');axis(hi,'equal')
    imagesc(seg.im,'Parent',hi);hold on
    colormap(hi,'gray');caxis([0,0.5]);
    plot(r_optim(:,1),r_optim(:,2),'+','LineWidth',2,'Color',cch,'Parent',hi);
    drawnow;
    
    cax = axes('Parent',hf);    
    linkaxes([cax,hi],'xy')
    contour(seg.fg - seg.bo,[0.2:0.1:0.6],'Parent',cax)
    cax.Position = hi.Position;
    axis(cax,'ij');
    cax.Visible='off';
    axis([cax,hi],'square')
    axis([cax,hi],'equal')
    drawnow
    
    
    %Draw final prediction boundaries:-------------------------------------
    hf = i*100+4;
    figure(hf),movegui(hf,'south');clf(hf)
    imshow(seg.im,[0 0.5]);hold on;
    plot(r_optim(:,1),r_optim(:,2),'+','LineWidth',2,'Color',cch);
    B2 = bwboundaries(seg.fg - seg.bo > 0.5);visboundaries(gca,B2,'color',[1 0. 0]);
    drawnow;
    %}
    
    %Data for difference threshold choices:
    for t = 1:numel(thr)
        bw_thr = (seg.fg-seg.bo)>thr(t);
        CC = bwconncomp(bw_thr);
        count_man(i) = size(r_optim,1);
        obj = cellfun(@numel,CC.PixelIdxList');
        count_alg(i,t) = sum(obj>0);
    end
    
    %{
    CC = bwconncomp((seg.fg-seg.bo)>0.5);
    Xx = nan(CC.NumObjects,1);
    Yy = nan(CC.NumObjects,1);
    for c = 1:numel(CC.PixelIdxList)
        [xcell,ycell] = ind2sub(size(seg.fg),CC.PixelIdxList{c});
        Xx(c) = mean(xcell);
        Yy(c) = mean(ycell);
    end
    
    Dd = ((Xx-r_optim(:,1)').^2+(Yy-r_optim(:,2)').^2).^0.5;
    Dd = Dd<dthr;
    
    
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
    disp(i)
end

   
diffcount = bsxfun(@minus,count_man,count_alg);
disp(diffcount);
counts = [count_man,count_alg];

figure(999);clf(999)
n = (1:5:2000)';
rootn = n.^0.5;
%errorbarpatch(n,n,rootn,[0.7 0.7 0.7],0.3);hold on
errorbarpatch(n,n,n/10,[0.7 0.7 0.7],0.3);hold on
plot(counts(1:numel(train_fileid),1),counts(1:numel(train_fileid),2),'o','MarkerSize',5,'MarkerFaceColor',[0 0.3 1],'MarkerEdgeColor',[0.7 0.7 0.7]);axis equal;
plot(counts(1:numel(val_fileid),1),counts(1:numel(val_fileid),2),'o','MarkerSize',5,'MarkerFaceColor',[1 0 0],'MarkerEdgeColor',[0.7 0.7 0.7]);axis equal;
%text(counts(1:numel(train_fileid),1),counts(1:numel(train_fileid),2),train_fileid,'Interpreter','None')
xlim([0 1700]);ylim([0 1700])
xlabel('Manual counts');
ylabel('Automated counts');
1;
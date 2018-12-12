load('TEST4.mat');
IM = IM./max(IM(:));

nhoodsize = 7;
padsize = [2*nhoodsize,2*nhoodsize];
IM = padarray(IM,padsize);
r = r+padsize;
%IM = imgaussfilt(IM,5);

%Pad the input with 2*nhood

figure(1);clf(1)
imagesc(IM);drawnow;hold on
colormap('gray')

xx = r(:,2);
yy = r(:,1);
plot(yy,xx,'sr','MarkerSize',20);

beta = 10;
nsteps = 10000;

cc = parula(nsteps);
sigma = 5;
c = 1./(2*pi*sigma^2).^0.5;
dfx = 1;
dfy = 1;

%Define neighbourhood
[nhoodX,nhoodY] = ndgrid(-nhoodsize:nhoodsize,-nhoodsize:nhoodsize);
nhoodX=nhoodX(:);
nhoodY=nhoodY(:);
rem = nhoodX.^2+nhoodY.^2>nhoodsize^2;
nhoodX(rem)=[];
nhoodY(rem)=[];
nhoodX = nhoodX';
nhoodY = nhoodY';
exit_steps = 0;
for n = 1:nsteps
    if max(abs(dfx(:)))+max(abs(dfy(:))) > 10^-10
        X = nhoodX+round(xx);
        Y = nhoodY+round(yy);
        ind = sub2ind(size(IM),X,Y);
        
        %delete(findall(1,'Tag','nhood'));
        %h = plot(Y,X,'oc','Tag','nhood');
        %pause(0.1)
        
        F   = c*sum(IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        dfx = c*sum(((X-xx)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        dfy = c*sum(((Y-yy)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        
        xx = xx + beta*dfx;
        yy = yy + beta*dfy;
        
        figure(1);
        plot(yy,xx,'.','Color',cc(n,:),'MarkerSize',20);hold on
        drawnow;
        
        if mod(n,100)==0
        disp([F,dfx,dfy])
        end
        exit_steps=n;
    end
    
end
plot(yy,xx,'sc','MarkerSize',20);

load('TEST.mat');
IM = IM./255;
%IM = imgaussfilt(IM,5);

figure(1);clf(1)
imagesc(IM);drawnow;hold on
colormap('gray')

xx = x(1);
yy = x(2);
plot(yy,xx,'sr','MarkerSize',20);

beta = 1000;
nsteps = 1000;

cc = parula(20);
nhoodsize = 5;
sigma = 5;
c = 1./(2*pi*sigma^2).^0.5;
dfx = 1;
dfy = 1;
for n = 1:nsteps
    if dfx+dfy > 10^-6
        X = nan((2*nhoodsize+1)^2,1);
        Y = nan((2*nhoodsize+1)^2,1);
        k = 1;
        for i = -nhoodsize:nhoodsize
            for j = -nhoodsize:nhoodsize
                X(k) = round(xx)+i;
                Y(k) = round(yy)+j;
                k = k+1;
            end
        end
        ind = sub2ind(size(IM),X,Y);
        
        delete(findall(1,'Tag','nhood'))
        %h = plot(Y,X,'oc','Tag','nhood');
        pause(0.1)
        
        F = (c/numel(ind))*sum(IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)));
        dfx = (c/numel(ind))*sum(((X-xx)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)));
        dfy = (c/numel(ind))*sum(((Y-yy)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)));
        
        xx = xx + beta*dfx;
        yy = yy + beta*dfy;
        
        figure(1);
        plot(yy,xx,'.','Color',cc(n,:),'MarkerSize',20);hold on
        drawnow;
        
        disp([F,dfx,dfy])
        exit_steps=n;
    end
    
end
plot(yy,xx,'sc','MarkerSize',20);

%This version implements a harmonic potential to prevent points from
%colliding.


function [final_pos] = optim_v4(IM,pos)
pos = unique(pos,'rows');
IM = IM./max(IM(:));

nhoodsize = 7;
padsize = [2*nhoodsize,2*nhoodsize];
IM = padarray(IM,padsize);
pos = pos+padsize;
%IM = imgaussfilt(IM,5);

xx = pos(:,2);
yy = pos(:,1);

%Pad the input with 2*nhood
figure(1);clf(1)
imagesc(IM);hold on
colormap('gray')
plot(yy,xx,'sr','MarkerSize',20);
drawnow

beta = 10;
beta_S = 10;
nsteps = 100;

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

cc = parula(nsteps);
for n = 1:nsteps
    if max(abs(dfx(:)))+max(abs(dfy(:))) > 10^-10
        x2x2 = (xx-xx').^2;
        y2y2 = (yy-yy').^2;
        
        x2x = xx-xx';
        y2y = yy-yy';
        
        r = (x2x2+y2y2).^0.5;
        
        X = nhoodX+round(xx);
        Y = nhoodY+round(yy);
        ind = sub2ind(size(IM),X,Y);
        
        Fi   = c*sum(IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        Fs   = -1*sum(S(r),2);
        
        [dsx,dsy] = dS(x2x,y2y,r);
        
        dfx = c*sum(((X-xx)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        dfy = c*sum(((Y-yy)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        
        dxx = beta*dfx - beta_S*dsx;
        dyy = beta*dfy - beta_S*dsy;
        
        maxjump = 1;
        dxx(abs(dxx)>maxjump) = sign(dxx(abs(dxx)>maxjump))*maxjump;
        dyy(abs(dyy)>maxjump) = sign(dyy(abs(dyy)>maxjump))*maxjump;
        
        xx = xx + dxx;
        yy = yy + dyy;
        
        
        if mod(n,100)==0
            disp([sum(Fi),sum(Fs)])
            %figure(1);
            %plot(yy,xx,'.','Color',cc(n,:),'MarkerSize',20);hold on
            %drawnow;
        end
        
        figure(1);
        plot(yy,xx,'.','Color',cc(n,:),'MarkerSize',20);hold on
        drawnow;
        exit_steps=n;
    end
    
end
plot(yy,xx,'sc','MarkerSize',20);
final_pos = [yy - padsize,xx - padsize];
end

function fx = S(r)
k = 0.1;
fx = 1./(k*(r.^2));
end

function [dsx,dsy] = dS(x2x,y2y,r)
k = 0.1;
r = r+10e10*eye(size(r));
r(r>10) = 10e10;
%r(r==0) = 0.1;

dsx = sum((1./k).*-1*r.^(-2).*2.*x2x,2);
dsy = sum((1./k).*-1*r.^(-2).*2.*y2y,2);
end


%This version implements a harmonic potential to prevent points from
%colliding.
T = load('TEST3.mat');
IM = T.IM;
pos = T.r;

pos = unique(pos,'rows');
IM = IM./max(IM(:));

nhoodsize = 7;
padsize = [2*nhoodsize,2*nhoodsize];
IM = padarray(IM,padsize);
pos = pos+padsize;
%IM = imgaussfilt(IM,5);

%Pad the input with 2*nhood
figure(1);clf(1)
imagesc(IM);drawnow;hold on
colormap('gray')

xx = pos(:,2);
yy = pos(:,1);
plot(yy,xx,'sr','MarkerSize',20);

beta = 1;
beta_S = 1;
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
        
        %delete(findall(1,'Tag','nhood'));
        %h = plot(Y,X,'oc','Tag','nhood');
        %pause(0.1)
        
        Fi   = c*sum(IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        Fs   = -1*sum(S(r),2);
        
        [dsx,dsy] = dS(x2x,y2y,r);
        
        dfx = c*sum(((X-xx)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        dfy = c*sum(((Y-yy)./sigma.^2).*IM(ind).*exp(-1.*(((Y-yy).^2+(X-xx).^2))./(2*sigma.^2)),2);
        
        dxx = beta*dfx - beta_S*dsx;
        dyy = beta*dfy - beta_S*dsy;
        
        maxjump = 0.5;
        dxx(abs(dxx)>maxjump) = sign(dxx(abs(dxx)>maxjump))*maxjump;
        dyy(abs(dyy)>maxjump) = sign(dyy(abs(dyy)>maxjump))*maxjump;
        
        xx = xx + dxx;
        yy = yy + dyy;
        figure(1);
        plot(yy,xx,'.','Color',cc(n,:),'MarkerSize',20);hold on
        drawnow;
        
        if mod(n,30)==0
            %disp([Fi,Fs,dfx,dfy])
            disp([sum(Fi),sum(Fs)])
        end
        exit_steps=n;
    end
    
end
plot(yy,xx,'sc','MarkerSize',20);

%{
function fx = S(r)
%Define k in terms of r0.
%x = 0:0.01:50;
%fx = @(x) -1./(1+exp(-.2.*(x)))+1;
%figure,plot(x,fx(x))
    k = 0.1;
    r = r+10e10*eye(size(r));
    fx = -1./(1+exp(-k.*(r)))+1;
end

function [dsx,dsy] = dS(x2x,y2y,r)
%Define k in terms of r0.
%x = 0:0.01:50;
%fx = @(x) -1./(1+exp(-.2.*(x)))+1;
%figure,plot(x,fx(x))
    k = 0.1;
    r = r+10e10*eye(size(r));
    r(r>50) = 10e10;
    r(r==0) = 0.1;
    
    dsx = (1+exp(-k.*r)).^-2 .* ...
        exp(-k.*r) .* ...
        -k*(1./r) .*...
        x2x;
    
    dsy = (1+exp(-k.*r)).^-2 .* ...
        exp(-k.*r) .* ...
        -k*(1./r) .*...
        y2y;
    
    dsx = sum(dsx,2);
    dsy = sum(dsy,2);
end
%}

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


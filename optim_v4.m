function [final_pos] = optim_v4(IM,init_pos,show_plots)
%This function optimizes in cell body center positions. Manually annotated
%cell body centers can be updated with this to use for downstream analysis.

%Params:-------------------------------------------------------------------
cellbodyradius = 7;
nhoodradius = cellbodyradius; % To evaluate intensity in neighborhood of the center.
far_thr = 1.5*cellbodyradius;     % Centers do not repel beyond this distance.

k = 0.1;         % Controls repulsion potential function
beta_G = 10;     % Step multiplier for intensity based update
beta_H = 10;     % Step multiplier for repulsion based update
maxsteps = 1000;
%--------------------------------------------------------------------------

init_pos = unique(init_pos,'rows');
IM = IM./max(IM(:));

%Image padding
padsize = [2*nhoodradius,2*nhoodradius];
IM = padarray(IM,padsize);
init_pos = init_pos+padsize;
x = init_pos(:,2);
y = init_pos(:,1);

if show_plots
    cc = parula(maxsteps);
    hf = figure;clf(hf)
    imagesc(IM);hold on
    colormap('gray')
    plot(y,x,'sr','MarkerSize',20);
    caxis([0 0.3])
    %Debug limits
    %xlim([1617,1838])
    %ylim([416,638])
    drawnow
end

%Define neighbourhood
[nhoodX,nhoodY] = ndgrid(-nhoodradius:nhoodradius,-nhoodradius:nhoodradius);
nhoodX=nhoodX(:)';
nhoodY=nhoodY(:)';
rem = nhoodX.^2+nhoodY.^2> nhoodradius^2;
nhoodX(rem)=[];
nhoodY(rem)=[];

%Initialize
step = 0;
not_converged = true;
not_exceedmaxsteps = true;

sigma = cellbodyradius/2;
c = 1./(2*pi*sigma^2).^0.5;
while not_exceedmaxsteps && not_converged
    %Note: implicit bsxfun assumed (R2016b+)!
    x2x = x-x';
    y2y = y-y';
    x2x2 = (x2x).^2;
    y2y2 = (y2y).^2;
    r = (x2x2+y2y2).^0.5;
    
    grid_x = nhoodX+round(x);
    grid_y = nhoodY+round(y);
    grid_ind = sub2ind(size(IM),grid_x,grid_y);
    
    I_gauss = IM(grid_ind).*exp(-1.*(((grid_y-y).^2+(grid_x-x).^2))./(2*sigma.^2));
    dG_x = c./sigma.^2*sum((grid_x-x).*I_gauss,2);
    dG_y = c./sigma.^2*sum((grid_y-y).*I_gauss,2);
    
    
    r_fixed = r+10e10*eye(size(r));
    r_fixed(r_fixed>far_thr) = 10e10;
    
    H = sum(1./(k*(r_fixed.^2)),2);
    dH_x = sum((1./k).*-1*r_fixed.^(-2).*2.*x2x,2);
    dH_y = sum((1./k).*-1*r_fixed.^(-2).*2.*y2y,2);
    
    dx = beta_G*dG_x - beta_H*dH_x;
    dy = beta_G*dG_y - beta_H*dH_y;
    
    %For stability
    maxjump = 1;
    dx(abs(dx)>maxjump) = sign(dx(abs(dx)>maxjump))*maxjump;
    dy(abs(dy)>maxjump) = sign(dy(abs(dy)>maxjump))*maxjump;
    
    x = x + dx;
    y = y + dy;
    
    if mod(step,10)==0
        G = c*sum(I_gauss,2);
        Fitness = sum(G - H);
        disp([Fitness, sum(G),sum(H)])
    end
    
    if show_plots
        figure(hf);
        plot(y,x,'.','Color',cc(step+1,:),'MarkerSize',20);hold on
        drawnow;
    end
    
    step = step+1;
    not_converged = max(abs(dG_x(:)))+max(abs(dG_y(:))) > 10^-10;
    not_exceedmaxsteps = step<maxsteps;
end

if not_converged
    disp('Did not converge')
    disp(['Exit at max_steps: ',num2str(step)])
else
    disp('Converged')
    disp(['Exit at step: ',num2str(step)])
end

if show_plots
    figure(hf)
    plot(y,x,'sc','MarkerSize',20);
    drawnow
end
final_pos = [y - padsize(2),x - padsize(1)];
end
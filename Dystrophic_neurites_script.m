load('Testcase.mat')
r = [Y,X,ones(size(X,1))];

%Filter image
gauss_sigma=1;
imfilt = imgaussfilt(IM./max(IM(:)),gauss_sigma);

%Solve the wave propagation equation, with speed of propogation
%proportional to the image intensity. FastMarching returns the time T at
%which the wave arrives at a given point, the Distance (~Speed/Time).
FM_max_dist=20;
[S,~,D,T]=FastMarchingTube(imfilt,r,FM_max_dist,[1,1,1]);

%***Contours to decide on thresholds 
figure,imshow(IM,[]);hold on;
c = contour(T,[10,20,25,30]);
d = contour(D,[1,3,5,7,9]);
B1 = bwboundaries(T<=500);
visboundaries(gca,B1,'Color','c');
drawnow

%For every 'object' normalize the time independently. This attempts to
%bring 'dim' objects at an equal footing weith bright objects. The
%thresholds for the labels are picked heuristically (hardcoded within)***
[Labels,Tadapt] = gen_Tadapt(IM,S,D,T);
figure,imshow(IM,[]);hold on;
c = contour(Tadapt,[0.8,1.5]);
drawnow

%Show the labels that mark 'inside' of the object
showlabel(IM,Labels>0)
plot(X,Y,'rx')
drawnow



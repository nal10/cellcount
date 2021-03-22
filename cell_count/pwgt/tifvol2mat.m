function IM = tifvol2mat(fname,start_ind,end_ind)
%start_ind and end_ind specify the starting and ending indices for planes 
%to load from a large tif file set these to [] to load all planes

InfoImage=imfinfo(fname);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);

NumberImages = round(NumberImages);
if isempty([start_ind,end_ind])
    start_ind=1;
    end_ind=NumberImages;
end

IM=zeros(nImage,mImage,end_ind-start_ind+1,'double');
TifLink = Tiff(fname, 'r');
i=1;
for t=start_ind:end_ind
    TifLink.setDirectory(t);
    IM(:,:,i)=TifLink.read();
    i=i+1;
end
TifLink.close();
end
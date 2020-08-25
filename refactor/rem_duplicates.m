function [i,j] = rem_duplicates(r,d_thr)

d = (((r(:,1)-r(:,1)').^2)+((r(:,2)-r(:,2)').^2)).^0.5;

%Remove diagonal entries
d = d+eye(size(d))*1000;

%Find indices that are close by
[i,j]=find(d<=d_thr);

ind = i<=j;
i = i(ind);
j = j(ind);
end
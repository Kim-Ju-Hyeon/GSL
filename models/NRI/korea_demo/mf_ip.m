function J=mf_ip(spikes)
N=size(spikes,2);
m=mean(spikes);
spikes=spikes-repmat(m, [size(spikes,1) 1]);
m=mean(spikes);
C=cov(spikes);
J=zeros(N);
for i=1:N
    for j=1:N
        J(i,j)=1/4*log(((1+m(i))*(1+m(j))+C(i,j))*((1-m(i))*(1-m(j))+C(i,j)))/...
            (((1+m(i))*(1-m(j))-C(i,j))*((1-m(i))*(1+m(j))-C(i,j)));
    end
end
end
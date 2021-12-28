function J=mf_tap(spikes)
    %% Preprocessing
    %spikes=binnedspikes(:,2:end);
    N=size(spikes,2);
    %spikes(spikes==0)=-1;
    m=mean(spikes);
    %spikes=spikes-repmat(m, [size(spikes,1) 1]);
    %m=mean(spikes);
    C=cov(spikes); 
    Cinv=inv(C);
    %% Calculate J
    J=zeros(N);
    for i=1:N
        for j=1:N
            J(i,j)=(sqrt(1-8*m(i)*m(j)*Cinv(i,j))-1)/(4*m(i)*m(j));
            if ~isreal(J(i,j))
                %J(i,j)
                J(i,j)=abs(J(i,j));
            end
        end
    end
    %J(logical(eye(size(J)))) = NaN;
end
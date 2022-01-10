function J=mf_sm(spikes)
    %% Preprocessing
%     addpath(genpath('L1GeneralExamples'));
%     spikes=standardizeCols(binnedspikes);
    m=mean(spikes);
    %spikes=spikes-repmat(m, [size(spikes,1) 1]);
    N=size(spikes,2);
    %spikes(spikes==0)=-1;
%     m=mean(spikes);
    C=cov(spikes); 
    %% Calculate J
    J=-inv(C)+mf_ip(spikes);
    for i=1:N
        for j=1:N
            J(i,j)=J(i,j)-C(i,j)/((1-m(i)^2)*(1-m(j)^2)-C(i,j)^2);
        end
    end
    %J(logical(eye(size(J)))) = NaN;
end
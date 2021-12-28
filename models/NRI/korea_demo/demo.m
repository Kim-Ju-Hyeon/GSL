%% generate grid cell network activity

% parameters
N=100;      % # of neurons
dt=.0001;   % time-step in s
tau=.01;    % synaptic time-constant in s

% weight parameters
sig_1=6.98;
sig_2=7;
a1=1;
a2=1.0005;

% create weight matrix
W=zeros(N);
for i=1:N
    for j=1:N
        x=min(abs(i-j),N-abs(i-j)); % distance between neurons i & j
        W(i,j)=a1*(exp(-(x)^2/(2*sig_1^2)) - a2*exp(-(x)^2/(2*sig_2^2)));
    end
end

% weight strength setting
load thresholds
idx=11;
r=thresholds(idx,1);			% recurrence strength
threshold=thresholds(idx,4);	% spiking threshold 2=ring, 4=LNP

b=0.001;				% uniform feedforward input
noise_sd=.3;			% amplitude of feedforward noise
noise_sparsity=1.5;		% noise is injected with the prob that a standard normal exceeds this
alpha=1e-3;

nsteps=4800000;				% no. of time-steps
niter=100;
s=zeros(1,N);			% initial activations
activations=nan(nsteps,N);	% store activations
spikes=nan(nsteps,N);	% spike raster
lambda=nan(nsteps,N);

act_tmp=nan(48000,N);
lam_tmp=nan(48000,N);
spk_tmp=nan(48000,N);

act_bin=nan(nsteps,N);
lam_bin=nan(nsteps,N);
spk_bin=nan(nsteps,N);

t=0;

for j=1:niter

    for i=1:nsteps
        % dynamics
        %I=r*s*W+(b*(1+noise_sd*randn(1,N).*(randn(1,N)>noise_sparsity))); %neuron inputs. The weight part is a simple matrix multiplication
        I=r*s*W+b; %if LNP
        %spike=I>threshold;  % binary neural spikes / ring
        spike=poissrnd(32*max(I,0)); %if LNP
        s=s+spike-s/tau*dt;

        spikes(i,:)=spike;
        %activations(i,:)=s; % if ring
        lambda(i,:)=32*max(I,0); % if LNP

        if mod(i,10000) == 0
            disp(i);
        end

        % plot
    %     subplot('Position',[.05 .55 .9 .4]);
    %     stem(I,'Marker','.','LineWidth',1);
    %     xlim([1 N]);
    %     ylim([-.002 .002]);
    %     refline(0,threshold);
    %     ylabel 'neural inputs g'
    %     title(sprintf('t = %.4f s',t))
    %     
    %     subplot('Position',[.05 .15 .9 .3]);
    %     area(s)
    %     xlim([1 N]);
    %     ylabel 'neural activations s'
    %     
    %     subplot('Position',[.04 .05 .92 .02])
    %     image(spike,'CDataMapping','scaled');
    %     colormap([1 1 1; .5 .7 1 ]);
    %     set(gca,'XGrid','on','XTick',0.5+1:N,'ticklength',[.01 .05],'YTick',[],'XTickLabel',[]);
    %     xlabel spikes
    %     
    %     drawnow

        t=t+dt;
    end
    
    for k=1:N
        if k == 1
            %act_tmp=activations(k:N:end, :);
            lam_tmp=lambda(k:N:end, :);
            spk_tmp=spikes(k:N:end, :);
        else
            %act_tmp=act_tmp+activations(i:N:end, :);
            lam_tmp=lam_tmp+lambda(k:N:end, :);
            spk_tmp=spk_tmp+spikes(k:N:end, :);
        end
    end
 
    activations=nan(nsteps,N);	% store activations
    spikes=nan(nsteps,N);	% spike raster
    lambda=nan(nsteps,N);
    
    % 1 juck = 48000 steps
    idx_in = ((j-1) * 48000) + 1;
    idx_out = j * 48000;
    
    %act_bin(idx_in:idx_out, :) = act_tmp;
    lam_bin(idx_in:idx_out, :) = lam_tmp;
    spk_bin(idx_in:idx_out, :) = spk_tmp;
end

%act_mean = zeros(100,nsteps/100,N);	% store activations
%spike_bin = zeros(100,nsteps/100,N); % spike raster
%lambda_bin = zeros(100,nsteps/100,N);

%for i=1:100
%    act_mean = act_mean + activations(i:100:end,:);
%    spike_bin = spike_bin + spikes(i:100:end,:);
%    lambda_bin = lambda_bin + lambda(i:100:end,:);
%end

%act_mean=act_mean/100;


%save('act_all.mat', 'activations', '-v7.3'); % If ring
%save('spk_all.mat', 'spikes', '-v7.3');
%save('lam_all.mat', 'lambda', '-v7.3'); % If LNP

%save('act_all.mat', 'activations', '-v7.3'); % If ring
save('spk_bin.mat', 'spk_bin', '-v7.3');
save('lam_bin.mat', 'lam_bin', '-v7.3'); % If LNP

%subplot(2,1,1); imagesc(1:100,(1:nsteps)*dt,spikes); title 'spikes'
%subplot(2,1,2); imagesc(1:100,(1:nsteps)*dt,activations); title 'activations'

% avg. ISI (s)
%N*t/sum(spikes(:))

%% Noise correlations
  
%C=corrcoef(spikes);
%C(logical(eye(N)))=nan;
%figure; imagesc(C); title 'noise correlations'
%abs_noise_corrs=abs(C(~tril(ones(size(C)))));
%figure; histogram(abs_noise_corrs); title 'abs. noise corr. histogram'

%r_sc=mean(abs_noise_corrs) % avg. abs. noise corr.

%% Inference using SM and TAP
%J_sm=mf_sm(spikes);
%J_sm(logical(eye(N)))=nan;
%figure; imagesc(J_sm)

%J_tap=mf_tap(spikes);
%J_tap(logical(eye(N)))=nan;
%figure; imagesc(J_tap)
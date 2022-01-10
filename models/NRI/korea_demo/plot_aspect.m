%imagesc(1:50,(1:4800000)*dt,lam_bin);
save('spk_bin.mat', 'spk_bin', '-v7.3');
save('lam_bin.mat', 'lam_bin', '-v7.3'); % If LNP
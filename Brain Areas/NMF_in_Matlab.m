clear

cd('../Results/lucy/150128/session01')

load coh_test

time = -0.65+0.25 : 0.025 : 1.75-0.25;
freq = 6:2:60;

[nt, nf, ntr, np] = size(coh);


X = permute(coh, [2 1 3 4]);
X = reshape(X, [nf nt*ntr*np]);

%--------------------------------------------------------------------------
% Time-Frequency domain NFM
%--------------------------------------------------------------------------
X = reshape(coh, [nt*nf ntr*np]);

K = 9;
[W, H, D] = nnmf(X, K);

figure
for n = 1:K
    subplot(3,3,n)
    imagesc(time, freq, reshape(W(:,n), nt, nf)')
    colormap jet
    xlabel('time (s)')
    ylabel('freq (Hz)')
    axis xy
    grid
end

K = 4;
[W, H, D] = nnmf(X, K);

% Plot time-frequency maps
figure
for n = 1:K
    subplot(2,2,n)
    imagesc(time, freq, reshape(W(:,n), nt, nf)')
    colormap jet
    xlabel('time (s)')
    ylabel('freq (Hz)')
    axis xy
    grid
end

%--------------------------------------------------------------------------
% Frequency domain NFM
%--------------------------------------------------------------------------
X = squeeze(mean(coh));
X = reshape(X, [nf ntr*np]);

K = 4;
[W, H, D] = nnmf(X, K, 'replicates', 25);

figure
for n = 1:K
    subplot(2,2,n)
    plot(freq, W(:,n))
    ylabel('Coh coeff')
    xlabel('Freq (Hz)')
    grid
end

K = 9;
[W, H, D] = nnmf(X, K, 'replicates', 25);

figure
for n = 1:K
    subplot(3,3,n)
    plot(freq, W(:,n))
    ylabel('Coh coeff')
    xlabel('Freq (Hz)')
    grid
end

%--------------------------------------------------------------------------
% Find K for NFM
%--------------------------------------------------------------------------

% Run NMF with varying number of clusters 
for k = 1:15
    [W, H, D] = nnmf(dmi, k, 'replicates', 25);
    R(:,k) = D;
end

% Find change point to find optimal number of clusters k (the elbow of the
% distribution)
figure
plot(R,'o-k')
grid
ylabel('RMS residual')
xlabel('K-dimensions (no. clusters)')

% Ask how many cluster
K = input('How many cluster?');

% Redo NMF with chosen number of clusters
[W, H, D] = nnmf(X, K, 'replicates', 25);

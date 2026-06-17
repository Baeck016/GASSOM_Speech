clear all; clc;

% load GASSOM model
% load("./modelset/TIMIT_onesentence/test_shuffle_tc_seed5/shuffle_tc20000/GASSOM_it=41.971k.mat");
load("./modelset/TIMIT_onesentence/shuffle_tc_seed5/mix_tc20000/GASSOM_it=41.971k.mat");
% load("./modelset/TIMIT_onesentence/shuttle_tc_seed5/SA2/mix_tc20000/GASSOM_it=34.295k.mat");
% load("./modelset/TIMIT_onesentence/shuttle_tc_seed5/random_1/mix_tc20000/GASSOM_it=36.899k.mat");
% load("./modelset/TIMIT_onesentence/shuttle_tc_seed5/random_2/mix_tc20000/GASSOM_it=36.597k.mat");

%% 
save_folder = "modelset/TIMIT_onesentence/shuttle_tc_seed5/mix_tc20000";

if ~exist(save_folder)
    mkdir(save_folder)
end

%% 
% Main code to run to cluster GASSOM Node
result = cluster_bases_hierarchical(obj, 0.4, 0.24, save_folder);   

%% 

cluster_shape = reshape(result.final_cluster, 10, 10)';
save(fullfile(save_folder, 'cluster_shape.mat'), 'cluster_shape');

cmap = [
        0.50 0.50 0.50;   % 0 -> Undecided (gray)
        0.00 0.00 1.00;   % 1 -> Male (blue)
        1.00 0.00 0.00;   % 2 -> Female (red)
        0.00 0.70 0.00;   % 3 -> Harmonic (green)
        1.00 0.50 0.00;   % 4 -> Baseless (orange)
        1.00 1.00 0.00    % 5 -> Silent (yellow)
    ];

figure('Visible','on');
imagesc(cluster_shape);
axis image;
colormap(cmap);
clim([-0.5 5.5]);

cb = colorbar;
cb.Ticks = 0:5;
cb.TickLabels = {'Undecided','Male','Female', 'Harmonic', 'Whisper', 'Silent'};
title("Cluster of nodes");
%}
%% 
silent_base(obj, 0.24, save_folder);

%%

% Identifying silent region replicating VAD method
function [silent_mask, thre_silent, cluster_indices, cluster_node_silent] = silent_base(obj, percentage, save_folder)
    cluster_node_silent = zeros(obj.n_subspace, 1);

    for p = 1:obj.n_subspace
        basis = obj.bases{1}(:, p);
        basis = reshape(basis, [800, 6]);
        power = basis(401:end, :).^2;
        cluster_node_silent(p,1) = median(mean(power, "all"));
    end

    thre_silent = prctile(cluster_node_silent, percentage);

    [cluster_indices, centroids] = kmeans(cluster_node_silent, 2, ...
        'Replicates', 20, ...
        'Distance', 'sqeuclidean');

    [~, silent_cluster_id] = min(centroids);
    silent_mask = (cluster_indices == silent_cluster_id);

    cluster_shape_silent = reshape(cluster_node_silent, obj.topo_subspace(1), obj.topo_subspace(2))';
    ClusterMap = reshape(cluster_indices, obj.topo_subspace(1), obj.topo_subspace(2))';

    figure;
    imagesc(ClusterMap);
    figure;
    heatmap(cluster_shape_silent);
    title("cluster of silent");
    saveas(gcf, save_folder + "/cluster_silent.png");
end

% Cluster whisper region using spectral tilt 
function [whisper_mask, whisper_cluster_full, tilt_all] = spectral_tilt_whisper(obj, remaining_mask)

    [n_rows, n_obs] = size(obj.bases{1});
    n_frames = 6;
    Nfft = n_rows / n_frames;
    fs = 16000;

    f = linspace(0, fs/2, Nfft).';
    fmin = 2000;
    fmax = 4000;

    tilt_all = NaN(n_obs,1);
    p_slope_all = NaN(n_obs,1);
    rho_all = NaN(n_obs,1);

    idx_remain = find(remaining_mask);

    for ii = 1:numel(idx_remain)
        p = idx_remain(ii);
        B = reshape(obj.bases{1}(:, p), [Nfft, n_frames]);
        mag = abs(B);
        Emean = mean(mag.^2, 2);

        [tilt_all(p), p_slope_all(p), rho_all(p), ~] = ...
            spectral_tilt_band_stats(f, Emean, fmin, fmax);
    end

    keep = remaining_mask & (p_slope_all < 0.05) & (abs(rho_all) >= 0.5) & ~isnan(tilt_all);

    whisper_mask = false(n_obs,1);
    whisper_cluster_full = zeros(n_obs,1);

    if any(keep)
        x_keep = tilt_all(keep);

        % k-means on raw tilt is easier to interpret than on zscore for centroid meaning
        [idx_cluster, C] = kmeans(x_keep, 2, 'Replicates', 20);

        % higher spectral tilt cluster = whisper-like cluster
        [~, whisper_cluster_id] = max(C);

        % store 1 = non-whisper, 2 = whisper for tested nodes
        tested_idx = find(keep);
        for i = 1:numel(tested_idx)
            p = tested_idx(i);
            if idx_cluster(i) == whisper_cluster_id
                whisper_cluster_full(p) = 2;
                
            else
                whisper_mask(p) = true;
                whisper_cluster_full(p) = 1;
            end
        end
    end

    mask_map = reshape(whisper_mask, obj.topo_subspace(1), obj.topo_subspace(2))';
    figure;
    imagesc(mask_map);
    title('Mask Whisper clusters');

    cluster_map = reshape(whisper_cluster_full, obj.topo_subspace(1), obj.topo_subspace(2))';
    figure;
    imagesc(cluster_map);
    title('Whisper clusters');
end  

% Function for identifying spectral tilt of each individual GASSOM node
function [slope_dB_per_oct, p_slope, rho, p_rho] = spectral_tilt_band_stats(f, E, fmin, fmax)
    f = f(:);
    E = E(:) + eps;

    mask = (f >= fmin) & (f <= fmax) & (f > 0);
    f_band = f(mask);
    E_band = E(mask);

    if numel(f_band) < 3
        slope_dB_per_oct = NaN;
        p_slope = NaN;
        rho = NaN;
        p_rho = NaN;
        return;
    end

    x = log2(f_band);
    y = 10*log10(E_band);

    mdl = fitlm(x, y);
    slope_dB_per_oct = mdl.Coefficients.Estimate(2);
    p_slope = mdl.Coefficients.pValue(2);

    [R, P] = corrcoef(x, y, 'Rows', 'complete');
    rho = R(1,2);
    p_rho = P(1,2);
end

function [higher_harmonic_mask, harmonic_score] = higher_harmonic_only(obj, remaining_mask, min_threshold)

    n_subspace = obj.n_subspace;
    higher_harmonic_mask = false(n_subspace, 1);
    harmonic_score = NaN(n_subspace, 1);

    for p = 1:n_subspace
        if ~remaining_mask(p)
            continue;
        end

        basis = obj.bases{1}(:, p);
        basis = reshape(basis, [800, 6]);

        amp = mean(abs(basis(401:end,:)), 2);
        amp = normalize(amp, "range");

        avg_baseless_range = mean(amp(150:end));
        avg_harmonic_range = mean(amp(13:100));

        harmonic_score(p) = avg_harmonic_range - avg_baseless_range;

        if avg_harmonic_range > min_threshold && avg_baseless_range <= min_threshold
            higher_harmonic_mask(p) = true;
        end
    end

    cluster_map = reshape(higher_harmonic_mask, obj.topo_subspace(1), obj.topo_subspace(2))';
    figure;
    imagesc(cluster_map);
    title('Higher harmonic mask');
    colorbar;
end

% Cluster Male and Female gender using F0
function f0 =  cepstrum_base_frequency(basis)
    basis   = basis(401:end, :);        
    
    power_basis = basis.^2;                       
    power_mean = mean(power_basis, 2) + eps;           
    
    % 3) Focus on low freqs (fundamental + a few harmonics)
    f     = (0:800/2)' * (16000/800);     % 0..8000, 401x1
    use   = (f >= 0 & f <= 1000);        % 0–1000 Hz
    Plow  = power_mean(use);
    figure('Visible','off');
    plot(Plow);
    set(gca, 'XTickLabel', get(gca,'XTick')*20);  % since df=20 Hz
    xlabel('Frequency (Hz)');
    ylabel('Power');
    title('Low-frequency power spectrum (0–1000 Hz)');
    xlim([0 length(Plow)-1]);
    grid on;

    Num_cep  = numel(Plow);

    %{
    figure('Visible','off');
    plot(log(Plow));
    set(gca, 'XTickLabel', get(gca,'XTick')*20);
    xlabel('Frequency (Hz)');
    ylabel('Log power');
    title('Log-magnitude spectrum (ripples every F_0)');
    xlim([0 length(Plow)-1]);
    grid on;
    %}
    
    % 4) Real cepstrum of log spectrum
    C = real(ifft(log(Plow), Num_cep));     % Ncep x 1
    
    % 5) Quefrency axis (seconds)
    dq = 1 / (Num_cep * 20);              % seconds
    q  = (0:Num_cep-1)' * dq;             % Ncep x 1
    
    % 6) Search for peak in plausible pitch period range (70–300 Hz)
    qMin = 1/300;                      % ≈ 3.3 ms
    qMax = 1/50;                       % ≈ 14.3 ms
    
    mask = (q >= qMin & q <= qMax);
    [~, relIdx] = max(C(mask));        % index inside mask
    absIdx = find(mask,1,'first') + relIdx - 1;

    T0 = q(absIdx);                    % seconds
    f0 = 1 / T0;                       % Hz fundamental frequency

    C_without = real(ifft((Plow), Num_cep)); 

    %{
    figure('Visible','off');
    plot(C_without);
    plot(q*1000, C, 'b-', 'LineWidth', 2);  % Convert q to ms!
    hold on;

    figure('Visible','off');
    plot(C);
    plot(q*1000, C, 'b-', 'LineWidth', 2);  % Convert q to ms!
    hold on;
    
    % Mark search window
    plot([qMin*1000 qMin*1000], ylim, 'r--', 'LineWidth', 2);  % qMin line
    plot([qMax*1000 qMax*1000], ylim, 'r--', 'LineWidth', 2);  % qMax line
    plot([q(absIdx)*1000 q(absIdx)*1000], ylim, 'g-', 'LineWidth', 3);  % YOUR PEAK!
    
    xlabel('Quefrency (ms)');
    ylabel('Cepstrum C[q]');
    title(sprintf('Cepstrum: Peak at T_0 = %.1f ms (f_0 = %.0f Hz)', T0*1000, f0));
    legend('C[q]', 'Search window 3.3–14.3 ms', sprintf('Peak T_0=%.1f ms', T0*1000), 'Location','best');
    xlim([0 50]);  % Up to 50 ms covers all plausible pitches
    grid on;
    %}
end

function [f0_mask, f0_cluster_full, f0_all] = cluster_f0_regions(obj, remaining_mask, save_folder)

    [~, n_obs] = size(obj.bases{1});
    f0_all = NaN(n_obs,1);

    idx_remain = find(remaining_mask);

    for ii = 1:numel(idx_remain)
        p = idx_remain(ii);
        basis = reshape(obj.bases{1}(:, p), [800, 6]);
        f0_all(p) = cepstrum_base_frequency(basis);
    end

    f0_shape = reshape(f0_all, 10, 10)';
    save(fullfile(save_folder, 'f0_shape.mat'), 'f0_shape');

    f0_mask = remaining_mask & ~isnan(f0_all);
    f0_cluster_full = zeros(n_obs,1);

    if any(f0_mask)
        x = zscore(f0_all(f0_mask));
        [idx_cluster, C] = kmeans(x, 2, 'Replicates', 20);

        [~, ord] = sort(C);
        newidx = zeros(size(idx_cluster));
        for k = 1:2
            newidx(idx_cluster == ord(k)) = k;
        end

        f0_cluster_full(f0_mask) = newidx;
    end

    cluster_map = reshape(f0_cluster_full, obj.topo_subspace(1), obj.topo_subspace(2))';
    figure;
    imagesc(cluster_map);
    title('F0 clusters');
end

% Cluster GASSOM node into silent, whisper, higher harmonic, male and female.
function result = cluster_bases_hierarchical(obj, percentage, min_threshold, save_folder)

    [silent_mask, thre_silent, silent_cluster, silent_score] = ...
        silent_base(obj, percentage, save_folder);

    remaining1 = ~silent_mask;

    [whisper_mask, whisper_cluster, tilt_all] = ...
        spectral_tilt_whisper(obj, remaining1);

    remaining2 = remaining1 & ~whisper_mask;

    [harmonic_mask, harmonic_score] = ...
        higher_harmonic_only(obj, remaining2, min_threshold);

    remaining3 = remaining2 & ~harmonic_mask;

    [f0_mask, f0_cluster, f0_all] = ...
        cluster_f0_regions(obj, remaining3, save_folder);

    result.silent_mask = silent_mask;
    result.whisper_mask = whisper_mask;
    result.harmonic_mask = harmonic_mask;
    result.f0_mask = f0_mask;

    result.thre_silent = thre_silent;
    result.silent_cluster = silent_cluster;
    result.whisper_cluster = whisper_cluster;
    result.f0_cluster = f0_cluster;

    result.silent_score = silent_score;
    result.tilt_all = tilt_all;
    result.harmonic_score = harmonic_score;
    result.f0_all = f0_all;

    final_cluster = zeros(obj.n_subspace, 1);   % undecided = 0
    
    final_cluster(f0_mask & (f0_cluster == 1)) = 1;   % male
    final_cluster(f0_mask & (f0_cluster == 2)) = 2;   % female
    
    % Higher harmonics
    final_cluster(harmonic_mask) = 3;
    
    % Whisper
    final_cluster(whisper_mask) = 4;
    
    % Silent
    final_cluster(silent_mask) = 5;
    result.final_cluster = final_cluster;
end

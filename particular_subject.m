clear all;
clc;


% Parameters for audio segments
fs = 16000; % sampling rate
segment_length = 0.075 * fs;
jump_length = round(0.075 / 5 * fs);

% Parameters for test sample
feature_type = 'abs_stft'; 
FFTL = 800; % STFT window length
OverlapRate = 0.9; % overlap rate

% Load GASSOM model
model = "./modelset/TIMIT_onesentence/Mix_Result/GASSOM_it=41.983k"; %10k
load(model)

directory = './dataset/TIMIT_onesentence/test_SA1_m'; 
file_pattern = fullfile(directory, '*wav');
m_audio_files = dir(file_pattern);

%% 

center_freq1 = 120;  % Hz
center_freq2 = 220;  % Hz
bandwidth = 3;      % ±50 Hz

% Create frequency axis for FFT (0 to fs/2)
freq_bins = linspace(0, fs/2, FFTL/2);  % 400 frequency bins

% Define Gaussian standard deviation
sigma = bandwidth / 2.355;

% Create Gaussian distributions across frequency bins
gaussian1 = exp(-((freq_bins - center_freq1).^2) / (2 * sigma^2));
gaussian1_rev = fliplr(gaussian1);
gaussian1 = [gaussian1_rev, gaussian1];
gaussian2 = exp(-((freq_bins - center_freq2).^2) / (2 * sigma^2));
gaussian2_rev = fliplr(gaussian2);
gaussian2 = [gaussian2_rev, gaussian2];

% Normalize Gaussian weights
% weight_sum = gaussian1 + gaussian2;
% w1_normalized = gaussian1 ./ weight_sum;
% w2_normalized = gaussian2 ./ weight_sum;
%% 


% Determine the minimum length of all audio files
min_len = inf;
for x = 1:length(m_audio_files)
    data = audioread(fullfile(directory, m_audio_files(x).name));
    min_len = min(min_len, length(data));
end

all_maxProj_node = cell(1, length(m_audio_files));
% all_maxProj_3_node = cell(1, length(m_audio_files));

for x = 1:length(m_audio_files)
    [signal, Fs] = audioread(fullfile(directory, m_audio_files(x).name));
    data = audioread(fullfile(directory, m_audio_files(x).name));
    data = data(1:min_len); % Truncate to the minimum length
    data = data ./ max(abs(data)); % Normalize
    len_data = length(data);

    maxProj_node_track = [];
    test_sample_collection=[];
    test_sample_est_collection=[];
    test_sample_before_reshape=[];
    k = 1; % Index inside data
    max_numsampl = round((len_data - segment_length) / jump_length);

    for i = 1:max_numsampl
        if k + segment_length - 1 > len_data % Zero-padding
            segment = zeros(segment_length, 1);
            segment(1:len_data + 1 - k) = data(k:len_data);
        else
            segment = data(k:k + segment_length - 1);
            k = k + jump_length;
        end
        
        switch feature_type
            case 'abs_fft'
                test_sample = abs(fft(segment));
                test_sample = test_sample(1:dim_patch_single);  
            case 'abs_stft'
                test_sample = abs(stft(segment, 'Window', hann(FFTL, 'periodic'), ...
                    'FFTLength', FFTL, 'OverlapLength', round(FFTL * OverlapRate)));
                test_sample = test_sample(:);
            case 'mfcc'
                test_sample = mfcc(segment, fs);
                test_sample = test_sample(2:end)';
        end

        % Normalize representation
        norm_coeff = norm(test_sample);
        test_sample = test_sample / norm_coeff;
        % test_sample_gau = reshape(test_sample,FFTL,[]);
        
        % test_sample_gau = test_sample_gau .* gaussian1';
        % test_sample_gau = test_sample_gau(:);
        
        [winners] = assomEncode(obj, test_sample); % Select the winner encode
        
        % Reconstruct by the node with maximal projection
        projection = obj.Proj;
        [~, maxProj_node] = max(obj.Proj); %either projection or nodeProb
        maxProj_node = winners;
        maxProj_node_track(i) = maxProj_node;
        
        % winner takes all
        test_sample_est= obj.bases{1}(:,maxProj_node) * obj.coef{1}(maxProj_node);

        test_sample=test_sample*norm_coeff;
        test_sample_est=test_sample_est*norm_coeff;

        test_sample_before_reshape = [test_sample_before_reshape test_sample];

        % reshape
        test_sample=reshape(test_sample,FFTL,[]);
        test_sample_est=reshape(test_sample_est,FFTL,[]);
       
        test_sample_collection=[test_sample_collection test_sample];
        test_sample_est_collection=[test_sample_est_collection test_sample_est];
    end
    
    all_maxProj_node{x} = maxProj_node_track;
    % all_maxProj_3_node{x} = maxProj_3_node_track;

    unique_node_numbers = unique(maxProj_node_track);
    freq_winning_node = histcounts(maxProj_node_track, [unique_node_numbers, max(unique_node_numbers) + 1]);

    % Frequency vector (for 800 frequency bins)
    F = linspace(1, fs/2, FFTL/2); % Frequencies from 0 to Nyquist frequency (fs/2)

    % Time vector (time frames)
    T = linspace(1, size(test_sample_est_collection,2), size(test_sample_est_collection,2)); % Adjust based on your segment duration

    % plot auditory spectrogram
    figure;
    subplot(4,1,1);
    plot(data);
    axis([0 length(data) -1 1])
    xlabel('time (second)')
    xticks([1 fs 2*fs 3*fs 4*fs]);
    xticklabels({'0', '1','2','3', '4'})
    ylabel('amplitude')
    title('Acoustic aveform of Mix')
    grid on;

    subplot(4,1,2);
    test_sample_collection = 10*log10(abs(test_sample_collection(FFTL/2:FFTL,:)));
    imagesc(T, F, test_sample_collection);
    title('Acoustic Spectrogram');
    xlabel('Frames');
    ylabel('Frequency (Hz)');
    ylim([0 500]);
    set(gca, 'YDir', 'normal');
    % saveas(gcf, fullfile('./modelset/TIMIT_onesentence/Single_Result/Energy_spectrogram/', [m_audio_files(x).name, '.jpg']));
    % 
    % plot reconstructed auditory spectrogram
    % subplot(4,1,3);
    % test_sample_est_collection = 10*log10(abs(test_sample_est_collection(FFTL/2:FFTL,:)));
    % imagesc(T, F, test_sample_est_collection);
    % title(sprintf('Reconstructed Spectrogram = %s', m_audio_files(x).name));
    % xlabel('Frames');
    % ylabel('Frequency (Hz)'); 
    % ylim([0 500]);
    % set(gca, 'YDir', 'normal');
    subplot(4,1,3);
    test_sample_est_collection = 10*log10(test_sample_est_collection(FFTL/2:FFTL,:));
    imagesc(T, F, test_sample_est_collection);
    title(sprintf('Reconstructed Spectrogram = %s', m_audio_files(x).name));
    xlabel('Frames');
    ylabel('Frequency (Hz)'); 
    ylim([0 500]);
    set(gca, 'YDir', 'normal');
    % saveas(gcf, fullfile('./modelset/TIMIT_onesentence/Single_Result/Energy_spectrogram/', [m_audio_files(x).name, '.jpg']));
    
    N  = numel(all_maxProj_node{1});
    t3 = linspace(T(1), T(end), N);

    subplot(4,1,4);
    plot(t3, all_maxProj_node{x});
    title('Plot of the best winning node')
    xlabel('Frames');
    xlim([T(1) T(end)]);

    %Plotting winning node for each sentence and speaker.
    % figure;
    % scatter(unique_node_numbers, freq_winning_node, 'filled');
    % xlabel('Winning Node');
    % ylabel(['Frequency in ', m_audio_files(x).name]);
    % title('Frequency of winning node');
    % grid on;
    % % Save the plot
    % saveas(gcf, fullfile('./modelset/TIMIT_onesentence/Single_Result/Winning_Node/', [m_audio_files(x).name, '.jpg']));
end
%% 
% Heatmap for winning node
Node_updateCount=zeros(100,1);

for i=1:100
    for k = 1: length(m_audio_files)
        Node_updateCount(i)= Node_updateCount(i) + sum(all_maxProj_node{k}==i);
    end
end

figure
bar(Node_updateCount);
xlabel('node index');
Node_updateCount_topo=reshape(Node_updateCount,10,10);
Node_updateCount_topo=Node_updateCount_topo'; 
figure
heatmap(Node_updateCount_topo);


%% 

% Calculate correlations and find the maximum and minimum
max_rho = -inf;
min_rho = inf;
max_pval = 0;
min_pval = 0;
max_pair = [];
min_pair = [];

correlation_results = []; % To hold correlation values
correlation_pvals = [];
file_pairs = {};          % To hold corresponding file names
pair_1 = {};
pair_2 = {};

for f1 = 1:length(m_audio_files) - 1
    for f2 = f1 + 1:length(m_audio_files)
        unique_numbers = unique([all_maxProj_node{f1}, all_maxProj_node{f2}]);

        freq_max_f1 = histcounts(all_maxProj_node{f1}, [unique_numbers, max(unique_numbers)+1]);
        freq_max_f2 = histcounts(all_maxProj_node{f2}, [unique_numbers, max(unique_numbers)+1]);
        
        [rho,pval] = corr(freq_max_f1', freq_max_f2'); %get the rho and pval

        % Store results
        correlation_results(end + 1) = rho;
        correlation_pvals(end + 1) = pval;
        file_pairs{end + 1} = {m_audio_files(f1).name, m_audio_files(f2).name};
        pair_1{end + 1} = m_audio_files(f1).name;
        pair_2{end + 1} = m_audio_files(f2).name;
        
        if rho > max_rho
            max_rho = rho;
            max_pval = pval;
            max_pair = [f1, f2];
        end
        
        if rho < min_rho
            min_rho = rho;
            min_pval = pval;
            min_pair = [f1, f2];
        end
    end
end



% Plot the frequencies for the pair with the greatest correlation
max_f1 = max_pair(1);
max_f2 = max_pair(2);

unique_numbers = unique([all_maxProj_node{max_f1}, all_maxProj_node{max_f2}]);

freq_max_f1 = histcounts(all_maxProj_node{max_f1}, [unique_numbers, max(unique_numbers)+1]);
freq_max_f2 = histcounts(all_maxProj_node{max_f2}, [unique_numbers, max(unique_numbers)+1]);

% figure;
% subplot(2, 1, 1);
% bar(freq_max_f1);
% xlabel('Number');
% ylabel('Frequency');
% title(['Histogram: ', m_audio_files(max_f1).name]);
% 
% subplot(2, 1, 2);
% bar(freq_max_f2);
% xlabel('Number');
% ylabel('Frequency');
% title(['Histogram: ', m_audio_files(max_f2).name]);

figure;
scatter(freq_max_f2, freq_max_f1, 'filled');
xlabel(['Frequency in ', m_audio_files(max_f2).name]);
ylabel(['Frequency in ', m_audio_files(max_f1).name]);
title('Frequency Comparison - Greatest Correlation');
grid on;

dx = 0.01*range(xlim);   % 2% of x-axis range

% Annotate the plot with unique numbers
for i = 1:length(unique_numbers)
    text(freq_max_f2(i) + dx, freq_max_f1(i), num2str(unique_numbers(i)));
end

% Add statistical information to the plot
text(1, max(max(freq_max_f1)), ['\rho = ', num2str(max_rho)]);
text(1, max(max(freq_max_f1)) - 1, ['p =', num2str(max_pval)]); 

% Print the correlation result
fprintf('Greatest correlation (\rho = %.2f) between files: %s and %s\n', max_rho, m_audio_files(max_f1).name, m_audio_files(max_f2).name);

% Plot the frequencies for the pair with the lowest correlation
min_f1 = min_pair(1);   
min_f2 = min_pair(2);
unique_numbers = unique([all_maxProj_node{min_f1}, all_maxProj_node{min_f2}]);

freq_min_f1 = histcounts(all_maxProj_node{min_f1}, [unique_numbers, max(unique_numbers)+1]);
freq_min_f2 = histcounts(all_maxProj_node{min_f2}, [unique_numbers, max(unique_numbers)+1]);

figure;
scatter(freq_min_f2, freq_min_f1, 'filled');
xlabel(['Frequency in ', m_audio_files(min_f2).name]);
ylabel(['Frequency in ', m_audio_files(min_f1).name]);
title('Frequency Comparison - Lowest Correlation');
grid on;

% Annotate the plot with unique numbers
for i = 1:length(unique_numbers)
    text(freq_min_f2(i) + dx, freq_min_f1(i), num2str(unique_numbers(i)));
end

% Add statistical information to the plot
text(1, max(max(freq_min_f1)), ['\rho = ', num2str(min_rho)]);
text(1, max(max(freq_min_f1)) - 1, ['p =', num2str(min_pval)]);

% % Print the correlation result
fprintf('Lowest correlation (\rho = %.2f) between files: %s and %s\n', min_rho, m_audio_files(min_f1).name, m_audio_files(min_f2).name);

% Combine results into a table for sorting
results_table = table(correlation_results',correlation_pvals' ,pair_1', pair_2');

% Sort the table in descending order
sorted_results = sortrows(results_table, "Var1", 'descend');

% Print sorted results
fprintf('Correlation Results (Descending Order):\n');
disp(sorted_results);

%% 
% Frequency vector (for 800 frequency bins)
F = linspace(1, fs/2, FFTL/2); % Frequencies from 0 to Nyquist frequency (fs/2)

%{

figure
for p=1:obj.n_subspace
    subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
    plot(abs(obj.bases{1}(:,p)));
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end
%% 

%}

figure
for p=1:obj.n_subspace
    subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
    basis = reshape(obj.bases{1}(:,p), [FFTL, 6]);
    basis = (abs(basis(1:FFTL/2,:)));
    imagesc(1:6, F, basis);
    set(gca, 'YDir', 'normal');
    % ylim([0 400]);
end



%}
%% 

[~, Proj_node] = max(obj.Proj);
disp(Proj_node);
coef = obj.bases{1}.*test_sample_gau;

figure
basis_1 = reshape(obj.bases{1}(:, 1), [FFTL, 6]);
basis_1 = 10*log10(abs(basis_1(FFTL/2+1:FFTL, :)));
imagesc(1:6, F, basis_1);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 500]);
title("Node 1")

figure
basis_51 = reshape(obj.bases{1}(:, 51), [FFTL, 6]);
basis_51 = 10*log10(abs(basis_51(FFTL/2+1:FFTL, :)));
imagesc(1:6, F, basis_51);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 500]);
title("Node 51")

figure
basis_81 = reshape(obj.bases{1}(:, 81), [FFTL, 6]);
basis_81 = 10*log10(abs(basis_81(FFTL/2+1:FFTL, :)));
imagesc(1:6, F, basis_81);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 500]);
title("Node 81")

figure
basis_91 = reshape(obj.bases{1}(:, 91), [FFTL, 6]);
basis_91 = 10*log10(abs(basis_91(FFTL/2+1:FFTL, :)));
imagesc(1:6, F, basis_91);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 500]);
title("Node 91")
%{

figure
basis_62 = reshape(obj.bases{1}(:, 62), [FFTL, 6]);
basis_62 = 10*log10(abs(basis_62(FFTL/2:FFTL, :)));
imagesc(1:6, F, basis_62);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 250]);
title("Node 62")

figure
basis_73 = reshape(obj.bases{1}(:, 73), [FFTL, 6]);
basis_73 = 10*log10(abs(basis_73(FFTL/2:FFTL, :)));
imagesc(1:6, F, basis_73);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
%ylim([0 250]);
title("Node 73")

figure
basis_73 = reshape(obj.bases{1}(:, 73), [FFTL, 6]);
basis_73 = 10*log10(abs(basis_73(FFTL/2:FFTL, :)));
imagesc(1:6, F, basis_73);
set(gca, 'YDir', 'normal');
ylabel("Frequency (Hz)");
ylim([0 250]);
title("Node 73")


%}


clear all; clc;

% load dataset
dataset_dir = './dataset/TIMIT_onesentence/test_SA1_f/';
listing = dir(dataset_dir);
listing = listing(~ismember({listing.name}, {'.', '..'}));
len_listing = length(listing);

% load GASSOM model
load("./modelset/TIMIT_onesentence/test_fade_seed5/GASSOM_it=41.971k.mat");

save_folder = "modelset/TIMIT_onesentence/test_fade_seed5/test_SA1_f/";
if ~exist(save_folder)
    mkdir(save_folder)
end

%%
encode_winner_count = zeros(obj.n_subspace, 1);
proj_winner_count = zeros(obj.n_subspace, 1);

tpb = textprogressbar(len_listing);
for i = 1:len_listing
    sample = audioread(sprintf(dataset_dir + "/" + listing(i).name));
    % add fade in and fade out if needed
    sample = fade(sample);
    
    [encode_winners_record, proj_winners_record, ~, ~] = test_sample(obj, sample);
    encode_winner_count = encode_winner_count + count_winner(obj, encode_winners_record);
    proj_winner_count = proj_winner_count + count_winner(obj, proj_winners_record);

    tpb(i);
end

[~, encode_winner_3_mode] = maxk(encode_winner_count, 3);
[~, proj_winner_3_mode] = maxk(proj_winner_count, 3);

%%
plot_bases(obj);

%%
plot_winner_heatmap(obj, encode_winner_count, "reconstruction", save_folder);
plot_winner_heatmap(obj, proj_winner_count, "projection", save_folder);
plot_mode_winner(obj, encode_winner_3_mode, "reconstruction", save_folder);
plot_mode_winner(obj, proj_winner_3_mode, "projection", save_folder);

%%
function y = fade (y)
    window_size = 0.075 * 16000;
    % apply rise-decay window
    rise_factor = (cos(linspace(pi / 2, 0, window_size)) .^ 2)';
    decay_factor = flip(rise_factor);
    y(1:length(rise_factor)) = y(1:length(rise_factor)) .* rise_factor;
    y(end - length(decay_factor) + 1:end) = y(end - length(decay_factor) + 1:end) .* decay_factor;
end

function [encode_winners_record, proj_winners_record, padded_audio_sample, train_sample_record] = test_sample (obj, sample)
    % parameter for audio segments
    fs=16000; % sampling rate
    segment_length=0.075*fs;
    jump_length=round(0.075/5*fs);

    % parameters for test sample
    FFTL=800; % STFT window length
    OverlapRate=0.9; % overlap rate

    train_sample_record = [];
    len_data=length(sample);
    k = 1;
    i = 1;

    while true
        if k+segment_length-1>len_data % zero-padding
            segment=zeros(segment_length,1);
            segment(1:len_data+1-k)=sample(k:len_data);
            % just want to ensure we dont make a sample only contains zero
            assert(any(segment ~= 0));

            padded_audio_sample = zeros(1, k + segment_length - 1);
            padded_audio_sample(1, 1:len_data) = sample;
        else
            segment=sample(k:k+segment_length-1);
        end

        train_sample=abs(stft(segment,fs,Window=hann(FFTL,'periodic'),...
        FFTLength=FFTL,OverlapLength=round(FFTL*OverlapRate)));
        train_sample=train_sample(:);
        train_sample_record = [train_sample_record train_sample];

        % normalize train_sample
        train_sample=train_sample/norm(train_sample);

        % record encode winner
        [encode_winners] = assomEncode(obj,train_sample);
        encode_winners_record(i) = encode_winners;

        % record projection winner
        [~, proj_winner] = max(obj.Proj);
        proj_winners_record(i) = proj_winner;

        if k+segment_length-1>len_data
            break;
        else
            k=k+jump_length;
            i = i + 1;
        end
    end
end

function ret = count_winner (obj, winners_record)
    ret = zeros(obj.n_subspace, 1);
    for i = 1:length(winners_record)
        ret(winners_record(i)) = ret(winners_record(i)) + 1;
    end
end

function plot_bases (obj)
    F = 16000/800 * (0:400);
    figure;
    for p=1:obj.n_subspace
        subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
        basis = obj.bases{1}(:, p);
        assert(all(basis >= 0) || all(basis <= 0));

        basis = reshape(basis, [800, 6]);
        basis = 10*log10(abs(basis(800/2:800, :)));
        imagesc(1:6, F, basis);
        set(gca, 'YDir', 'normal');
        ylim([0 440]);
    end
    figure;
    for p=1:obj.n_subspace
        subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
        basis = obj.bases{1}(:, p);
        assert(all(basis >= 0) || all(basis <= 0));

        basis = reshape(basis, [800, 6]);
        amp = mean(abs(basis(400:end,:)), 2);
        amp = normalize(amp, "range");

        plot(F, amp);
        xlim([0 440]);
        ylim([0 1]);
        hold on;
        line([120 120], ylim);
        line([220 220], ylim);
    end
end

function plot_winner_heatmap (obj, winner_count, winner_metric, save_folder)
    fig = figure;
    theme(fig, "light");
    h = heatmap(reshape(winner_count, obj.topo_subspace(1), obj.topo_subspace(2))');
    h.XDisplayLabels = 1:10;
    h.YDisplayLabels = 0:10:90;
    title(sprintf("winner distribution (" + winner_metric + ")"));
    saveas(gcf, sprintf(save_folder + "/winner_heatmap_" + winner_metric + ".png"));
end

function plot_mode_winner (obj, modes, winner_metric, save_folder)
    F = 16000/800 * (0:400);
    k = length(modes);
    winners_str = "";

    fig = figure;
    theme(fig, "light");

    for i = 1:k
        winners_str = winners_str + modes(i) + " ";

        basis = obj.bases{1}(:, modes(i));
        basis = reshape(basis, [800, 6]);
        amp = mean(abs(basis(400:end,:)), 2);

        subplot(k, 2 , 2 * i - 1);
        basis = 10*log10(abs(basis(800/2:800, :)));
        imagesc(1:6, F, basis);
        set(gca, 'YDir', 'normal');
        ylim([0 440]);

        subplot(k, 2, 2 * i);
        amp = normalize(amp, "range");
        plot(F, amp);
        xlim([0 440]);
        ylim([0 1]);
        hold on;
        line([120 120], ylim);
        line([220 220], ylim);
    end

    sgtitle(sprintf("top " + k + " winner (" + winner_metric + ")\n" + winners_str));

    saveas(gcf, sprintf(save_folder + "/top_" + k + "_winners_" + winner_metric + ".png"));
end

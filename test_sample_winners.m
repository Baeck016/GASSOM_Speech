clear all; clc;

% load GASSOM model
load("./modelset/TIMIT_onesentence/test_shuffle_tc_seed5/shuffle_tc20000/GASSOM_it=41.971k.mat");

% sample = audioread("dataset/TIMIT_onesentence/test_SA1/SA1_FAEM0.WAV");
sample = audioread("dataset/TIMIT_onesentence/test/CombinedFiles/FAKS0_SI1573.WAV.wav");

save_folder = "modelset/TIMIT_onesentence/test_shuffle_tc_seed5/shuffle_tc20000/FAKS0_SI1573";

if ~exist(save_folder)
    mkdir(save_folder)
end

%%
% plot_bases(obj);

%%
% add fade in and fade out if needed
% sample = fade(sample);

[encode_winners_record, proj_winners_record, padded_audio_sample, test_sample_record] = test_sample(obj, sample, true, save_folder);
encode_winner_count = count_winner(obj, encode_winners_record);
proj_winner_count = count_winner(obj, proj_winners_record);
[~, encode_winner_3_mode] = maxk(encode_winner_count, 3);
[~, proj_winner_3_mode] = maxk(proj_winner_count, 3);

%%
plot_winner_heatmap(obj, encode_winner_count, "reconstruction", save_folder);
% plot_winner_heatmap(obj, proj_winner_count, "projection", save_folder);
plot_mode_winner(obj, encode_winner_3_mode, "reconstruction", save_folder);
% plot_mode_winner(obj, proj_winner_3_mode, "projection", save_folder);

%%
plot_winner_track( ...
    padded_audio_sample, ...    
    encode_winners_record, encode_winner_3_mode, ...
    proj_winners_record, proj_winner_3_mode, ...
    test_sample_record, ...
    save_folder ...
);

%%
save(sprintf(save_folder + "/variables.mat"), ...
    "encode_winners_record", "padded_audio_sample", "test_sample_record", ...
    "encode_winner_count", "encode_winner_3_mode" ...
);

beep;

%%
function y = fade (y)
    window_size = 0.075 * 16000;
    % apply rise-decay window
    rise_factor = (cos(linspace(pi / 2, 0, window_size)) .^ 2)';
    decay_factor = flip(rise_factor);
    y(1:length(rise_factor)) = y(1:length(rise_factor)) .* rise_factor;
    y(end - length(decay_factor) + 1:end) = y(end - length(decay_factor) + 1:end) .* decay_factor;
end

function [encode_winners_record, proj_winners_record, padded_audio_sample, test_sample_record] = test_sample (...
    obj, sample, do_track_procces, save_folder ...
)
    if nargin < 3
        do_track_procces = false;
    end

    if do_track_procces
        frame_record_folder = sprintf("%s/frame_records", save_folder);
        if ~exist(frame_record_folder)
            mkdir(frame_record_folder)
        end
    end

    % parameter for audio segments
    fs=16000; % sampling rate
    segment_length=0.075*fs;
    jump_length=round(0.075/5*fs);

    % parameters for test sample
    FFTL=800; % STFT window length
    OverlapRate=0.9; % overlap rate

    test_sample_record = [];
    len_data=length(sample);
    k = 1;
    i = 1;

    tbp = textprogressbar(len_data, "showremtime", true);

    % pre-allocated memory to store record
    mae_record = zeros(500,100);
    nodeprobTmp_record = zeros(500,100);
    proj_record = zeros(500,100);
    nodeprobTmp_rank_record = zeros(500,100);

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

        test_sample_stft=abs(stft(segment,fs,Window=hann(FFTL,'periodic'),...
        FFTLength=FFTL,OverlapLength=round(FFTL*OverlapRate)));

        test_sample=test_sample_stft(:);
        test_sample_record = [test_sample_record test_sample];

        % normalize train_sample
        test_sample=test_sample/norm(test_sample);

        mae = mean(abs(test_sample - abs(obj.bases{1})));
        [~, mae_winner] = min(mae);
        mae_record(i,:) = mae;

        % record encode winner
        [encode_winners] = assomEncode(obj,test_sample);
        encode_winners_record(i) = encode_winners;
        nodeprobTmp_record(i,:) = obj.nodeprobTmp;
        nodeprobTmp_rank_record(i,:) = 101 - tiedrank(obj.nodeprobTmp);

        assert(nodeprobTmp_rank_record(i,encode_winners) == 1);
        
        % record projection winner
        [~, proj_winner] = max(obj.Proj);
        proj_winners_record(i) = proj_winner;
        proj_record(i,:) = obj.Proj;

        if do_track_procces
            F = 16000/800 * (0:400);

            % -- frame record
            % fig = figure;
            fig = figure("visible", "off");
            theme(fig, "light");
            tiledlayout(2,2,"TileSpacing","compact","Padding","compact");
            % -- frame
            nexttile;
            imagesc(1:6, F, 10 * log10(test_sample_stft(800/2:800,:)));
            % imagesc(1:6, F, test_sample_stft(800/2:800,:));
            title(sprintf("frame %d (0Hz-8000Hz)", i));
            set(gca, 'YDir', 'normal');
            nexttile;
            imagesc(1:6, F, 10 * log10(test_sample_stft(800/2:800,:)));
            % imagesc(1:6, F, test_sample_stft(800/2:800,:));
            title(sprintf("frame %d (0Hz-1500Hz)", i));
            set(gca, 'YDir', 'normal');
            ylim([0 1500]);
            % -- nodeprobTmp
            nexttile;
            plot_heatmap(reshape(obj.nodeprobTmp, 10, 10)');
            title(sprintf("reconstruction values (winner %d)", encode_winners));
            % -- projection
            nexttile;
            plot_heatmap(reshape(obj.Proj, 10, 10)');
            title(sprintf("projection values (winner %d)", proj_winner));
            % -- save
            set(gcf, 'WindowState', 'maximized');
            saveas(gcf, sprintf("%s/frame%d_rec%d_proj%d.png", frame_record_folder, i, encode_winners, proj_winner));
            close;

            % mae
            fig = figure("visible", "off");
            theme(fig, "light");
            plot_heatmap(reshape(mae,10,10)');
            title(sprintf("frame%d MAEs (winner %d)", i, mae_winner));
            set(gcf, 'WindowState', 'maximized');
            saveas(gcf, sprintf("%s/frame%d_mae%d.png", frame_record_folder, i, mae_winner));
            close;
        end

        if k+segment_length-1>len_data
            break;
        else
            k=k+jump_length;
            i = i + 1;
        end

        tbp(k);
    end

    % plot means and stds
    nodeprobTmp_record = nodeprobTmp_record(1:i,:);
    nodeprobTmp_rank_record = nodeprobTmp_rank_record(1:i,:);
    proj_record = proj_record(1:i,:);
    mae_record = mae_record(1:i,:);

    plot_mean_std(nodeprobTmp_record, "reconstruction");
    plot_mean_std(nodeprobTmp_rank_record, "reconstruction rank");
    plot_mean_std(proj_record, "projection");
    plot_mean_std(mae_record, "mae");

    function plot_heatmap (data)
        h = heatmap(data);
        h.ColorbarVisible = "off";
        h.XDisplayLabels = 1:10;
        h.YDisplayLabels = 0:10:90;
    end

    function plot_mean_std (data, name)
        fig = figure("visible", "off");
        theme(fig, "light");
        tiledlayout(1,2,"TileSpacing","compact","Padding","compact");
        nexttile;
        plot_heatmap(reshape(mean(data), 10, 10)');
        title("mean");
        nexttile;
        plot_heatmap(reshape(std(data), 10, 10)');
        title("std");
        sgtitle(sprintf("%s value", name));
        set(gcf, 'WindowState', 'maximized');
        saveas(gcf, sprintf("%s/%s_mean_std.png", save_folder, name));
        close;
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
        % basis = 10*log10(abs(basis(800/2:800, :)));
        basis = abs(basis(800/2:800, :));
        imagesc(1:6, F, basis);
        set(gca, 'YDir', 'normal');
        ylim([0 1500]);
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
        xlim([0 1500]);
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
        % basis = basis(800/2:800, :);
        basis = 10*log10(abs(basis(800/2:800, :)));
        % basis = abs(basis(800/2:800, :));
        imagesc(1:6, F, basis);
        set(gca, 'YDir', 'normal');
        ylim([0 1500]);

        subplot(k, 2, 2 * i);
        amp = normalize(amp, "range");
        plot(F, amp);
        xlim([0 1500]);
        ylim([0 1]);
        hold on;
        line([120 120], ylim);
        line([220 220], ylim);
    end

    sgtitle(sprintf("top " + k + " winner (" + winner_metric + ")\n" + winners_str));

    saveas(gcf, sprintf(save_folder + "/top_" + k + "_winners_" + winner_metric + ".png"));
end

function plot_winner_track ( ...
    sample, ...
    encode_winners_record, encode_winner_modes, ...
    proj_winners_record, proj_winner_modes, ...
    train_sample_record, ...
    save_folder ...
)
    fs = 16000;
    len_data = length(sample);
    FFTL = 800;

    mode_k = length(encode_winner_modes);
    assert(mode_k == length(proj_winner_modes))

    fig = figure;
    theme(fig, "light");
   
    % Plot the speech amplitude time history
    subplot(4,1,1);
    plot(sample);
    xlim([0 len_data]);
    xlabel('time (second)')
    xticks([fs 2*fs 3*fs 4*fs]);
    xticklabels({'1', '2', '3', '4'})
    ylabel('amplitude')
    title('Acoustic Waveform')
    % grid on;    

    % Plot the winning node number as a function of time
    subplot(4,1,2);
    plot(encode_winners_record, "DisplayName", "winner track");
    xlim([1 length(encode_winners_record)]);
    for i = 1:mode_k
        line(xlim, [encode_winner_modes(i) encode_winner_modes(i)], "DisplayName", "mode " + i);
    end
    ylim([-5 105]);
    title("winner tracking (reconstruction)");
    legend("show");

    % 
    subplot(4,1,3);
    plot(proj_winners_record, "DisplayName", "winner track");
    xlim([1 length(proj_winners_record)]);
    for i = 1:mode_k
        line(xlim, [proj_winner_modes(i) proj_winner_modes(i)], "DisplayName", "mode " + i);
    end
    ylim([-5 105]);
    title("winner tracking (projection)");
    legend("show");

    % Plot the acoustic spectrogram
    subplot(4,1,4);
    imagesc(10 * log10(train_sample_record(1:FFTL/2,:)));
    title('Acoustic Spectrogram');
    xlabel('Frame');
    % xticks([1, 201, 401, 601 801]);
    % xticklabels({'0', '1','2','3', '4'});
    ylabel('Frequency (Hz)');
    yticks([1 100 200 300 400]);
    yticklabels({'8000','6000','4000','2000','0'});

    saveas(gcf, sprintf(save_folder + "/winner_tracking.png"));
end

clear all; clc;

% result directory
% result_dir='modelset/TIMIT_onesentence/Mix_Result'; 
result_dir='modelset/TIMIT_onesentence/test_shuffle_tc_seed5/MF_tc20000'; 
if ~exist(result_dir, 'dir')
    mkdir(result_dir)
end

%% initiate GASSOM
% parameter for GASSOM
dim_patch_single=4800; % "dim_patch_single" = total dimensions of "train_sample"
topo_subspace=[10 10]; % 10 X 10 nodes
max_iter=5e5; % max iteration for training ASSOM

PARAM{1}=dim_patch_single;
PARAM{2}=topo_subspace;
PARAM{3}=max_iter;

% set random seed
rng(5);
obj=GASSOM_Online(PARAM);
pre_bases=obj.bases;

%% training
% parameter for audio segments
fs=16000; % sampling rate
segment_length=0.075*fs; % 3/40 * 16000Hz
jump_length=round(0.075/5*fs); % 3/(40 * 5) * 16000Hz*

% parameters for training sample
FFTL=800; % STFT window length
OverlapRate=0.9; % overlap rate
feature_type='abs_stft'; 

% add dataset
dataset_dir = './dataset/TIMIT_onesentence/train_SA1'; % this is a very small dataset %dataset segregated & data not segregated
listing = dir(dataset_dir); %
% get info of dataset
listing = listing(~ismember({listing.name}, {'.', '..'}));
len_listing = length(listing);
% Randomize the data for training
% shuffle_indices = randperm(len_listing);
% listing = listing(shuffle_indices, :);
listing = listing(len_listing:-1:1, :);

train_sample_collection=[];
pre_bases={};

i=1; % pre_bases index
j=1; % index of data
k=1; % index inside data
do_fade = false;

%initialize the first audio dataset
data = audioread([dataset_dir '/' listing(j).name]);
if do_fade
    data = fade(data);
end
len_data=length(data);

hw=waitbar(0,'Running...'); % create a waitbar running
tpb = textprogressbar(len_listing, "showremtime", true);
for it=1:obj.max_iter

    if k+segment_length-1>len_data % zero-padding
        segment=zeros(segment_length,1);
        segment(1:len_data+1-k)=data(k:len_data);
        % just want to ensure we dont make a sample only contains zero
        assert(any(segment ~= 0));

        tpb(j);
        j=j+1; 
        if j>length(listing) % already trained through the whole dataset
            save([result_dir '/GASSOM_it=' num2str(it/1000) 'k.mat'],'obj');
            break
        end
        data=audioread([dataset_dir '/' listing(j).name]); % load a new audio data
        if do_fade
            data = fade(data);
        end
        len_data=length(data);
        k=1;
    else
        segment=data(k:k+segment_length-1);
        k=k+jump_length;
    end

    switch feature_type
        case 'abs_fft'
            train_sample=abs(fft(segment));
            train_sample=train_sample(1:dim_patch_single);
        case 'abs_stft'
            train_sample=abs(stft(segment,fs,Window=hann(FFTL,'periodic'),...
            FFTLength=FFTL,OverlapLength=round(FFTL*OverlapRate)));

            % TODO: this line consumes huge computation power
            % train_sample_collection = [train_sample_collection train_sample];

            train_sample=train_sample(:);
        case 'waveform'
            train_sample=segment;
    end

    % normalize train_sample
    train_sample=train_sample/norm(train_sample);
    % 
    [winners] = assomEncode(obj,train_sample); % select the winner encode
    updateBasis(obj,train_sample); % update bases of GASSOM
    winners_record(it)=winners; % record winning node of each iteration

    % record train history
    % new_bases=obj.bases;
    % loss(it)=norm(pre_bases{1}-new_bases{1})+norm(pre_bases{2}-new_bases{2}); % record loss after each iteration
    % train_sample_est=reconstruction(obj,train_sample,winners); % reconstruction by winner node
    % recon_error(it)=norm(train_sample_est-train_sample); % record reconstruction error of each iteration
    % if mod(it, 100) == 0
    %     pre_bases{i}=new_bases;
    %     i = i + 1;
    % end
    if round(it/10000)==it/10000 % save model every 4k it
        save([result_dir '/GASSOM_it=' num2str(it/1000) 'k.mat'],'obj');
    end

    waitbar(it/obj.max_iter,hw);
end
close(hw)
beep; % notify me training is ended

function y = fade (y)
    window_size = 0.075 * 16000;
    % apply rise-decay window
    rise_factor = (cos(linspace(pi / 2, 0, window_size)) .^ 2)';
    decay_factor = flip(rise_factor);
    y(1:length(rise_factor)) = y(1:length(rise_factor)) .* rise_factor;
    y(end - length(decay_factor) + 1:end) = y(end - length(decay_factor) + 1:end) .* decay_factor;
end

%% save training result
%{
fig = figure('Visible','off');
pre_bases_sample = 10*log10(abs(pre_bases(FFTL/2:FFTL,:)));
imagesc(T, F, pre_bases_sample);
title('Acoustic Spectrogram');
xlabel('Frames');
ylabel('Frequency (Hz)');
set(gca, 'YDir', 'normal');
save_path = fullfile(result_dir, 'pre_bases.jpg');

% Save the figure
saveas(gcf, save_path);
%}

% save([result_dir '/GASSOM_prebase.mat'], 'pre_bases')

%% plot histgram and heatmap of winning node
Node_updateCount=zeros(obj.n_subspace,1);
for i=1:obj.n_subspace
    Node_updateCount(i)=sum(winners_record==i);
end
figure;
bar(Node_updateCount);
xlabel('node index');
saveas(gcf, [result_dir '/hist_updated_node.jpg'])
Node_updateCount_topo=reshape(Node_updateCount,10,10);
Node_updateCount_topo=Node_updateCount_topo'; 
figure;
heatmap(Node_updateCount_topo);
saveas(gcf, [result_dir '/heatmap_updated_node.jpg'])

%% plot pre base
% Frequency vector (for 800 frequency bins)
% Frequencies from 0 to Nyquist frequency (fs/2)
% F = linspace(1, 16000/2, 800/2);
F = 16000/800 * (0:400); % from matlab fft webpage

% figure;
% for p=1:obj.n_subspace
%     subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
%     basis = pre_bases(:, p);
%     % just make sure all number in the basis share same sign for the abs operation
%     assert(all(basis >= 0) || all(basis <= 0));

%     basis = reshape(basis, [800, 6]);
%     basis = 10*log10(abs(basis(800/2:800, :)));
%     imagesc(1:6, F, basis);
%     set(gca, 'YDir', 'normal');
%     ylim([0 440]);
% end

%% plot bases
% clear all;

% result_dir='modelset/TIMIT_onesentence/Result';
% obj = load(result_dir + "/GASSOM_it=103.365k.mat").obj;

F = 16000/800 * (0:400);
% figure;
fig = figure("visible", "off");
theme(fig, "light")
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
saveas(gcf, result_dir + "/bases.png");

%% plot bases with average amplitude
fig = figure("visible", "off");
theme(fig, "light")
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
saveas(gcf, result_dir + "/bases_mean_amp.png");

%% plot collected train samples
% figure;
% train_sample = 10*log10(abs(train_sample_collection(FFTL/2:FFTL,:)));
% imagesc(T, F, train_sample);
% title('Acoustic Spectrogram');
% xlabel('Frames');
% ylabel('Frequency (Hz)');
% set(gca, 'YDir', 'normal');

clear all
clc
% parameter for audio segments
fs=16000; % sampling rate
segment_length=0.075*fs; % 3/40 * 16000Hz
jump_length=round(0.075/5*fs); % 3/(40 * 5) * 16000Hz*

% parameters for training sample
FFTL=800; % STFT window length
OverlapRate=0.9; % overlap rate
feature_type='abs_stft'; 

% parameter for GASSOM
dim_patch_single=4800; % "dim_patch_single" = total dimensions of "train_sample"
topo_subspace=[10 10]; % 10 X 10 nodes
max_iter=5e5; % max iteration for training ASSOM


% initiate GASSOM
PARAM{1}=dim_patch_single;
PARAM{2}=topo_subspace;
PARAM{3}=max_iter;
obj=GASSOM_Online(PARAM);
pre_bases=obj.bases;

% add dataset
dataset_dir = './dataset/TIMIT_onesentence/train_SA1'; % this is a very small dataset %dataset segregated & data not segregated
listing = dir(dataset_dir); %
% get info of dataset
listing = listing(~ismember({listing.name}, {'.', '..'}));
len_listing = length(listing);

% result directory
result_dir='modelset/TIMIT_onesentence/Mix_Result'; 
if ~exist(result_dir, 'dir')
    mkdir(result_dir)
end

j=1; % index of data
k=1; % index inside data

%initialize the first audio dataset
data = audioread([dataset_dir '/' listing(j).name]);
len_data=length(data);

%% 
%Randomize the data for training

shuffle_indices = randperm(len_listing);
listing = listing(shuffle_indices, :);


%%
train_sample_collection=[];
pre_bases={};
i=1;

hw=waitbar(0,'Running...'); % create a waitbar running
for it=1:max_iter

    if k+segment_length-1>len_data % zero-padding
        segment=zeros(segment_length,1);
        segment(1:len_data+1-k)=data(k:len_data);
        j=j+1; 
        if j>length(listing) % already trained through the whole dataset
            save([result_dir '/GASSOM_it=' num2str(it/1000) 'k.mat'],'obj');
            break
        end
        data=audioread([dataset_dir '/' listing(j).name]); % load a new audio data
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
            train_sample_collection = [train_sample_collection train_sample];
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
    new_bases=obj.bases;
    %loss(it)=norm(pre_bases{1}-new_bases{1})+norm(pre_bases{2}-new_bases{2}); % record loss after each iteration
    train_sample_est=reconstruction(obj,train_sample,winners); % reconstruction by winner node
    recon_error(it)=norm(train_sample_est-train_sample); % record reconstruction error of each iteration
    if mod(it, 100) == 0
        pre_bases{i}=new_bases;
        i = i + 1;
    end
    if round(it/10000)==it/10000 % save model every 4k it
        save([result_dir '/GASSOM_it=' num2str(it/1000) 'k.mat'],'obj');
    end
    waitbar(it/max_iter,hw);
end
close(hw)

%% 
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

save([result_dir '/GASSOM_prebase.mat'], 'pre_bases')

%%
% plot histgram and heatmap of winning node
Node_updateCount=zeros(100,1);
for i=1:100
    Node_updateCount(i)=sum(winners_record==i);
end
figure
bar(Node_updateCount);
xlabel('node index');
saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/hist_updated_node.jpg')
Node_updateCount_topo=reshape(Node_updateCount,10,10);
Node_updateCount_topo=Node_updateCount_topo'; 
figure
heatmap(Node_updateCount_topo);
saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/heatmap_updated_node.jpg')
%%
% Frequency vector (for 800 frequency bins)
F = linspace(1, 16000/2, 800/2); % Frequencies from 0 to Nyquist frequency (fs/2)

figure
for p=1:100
    subplot(10,10,p);
    basis = reshape(pre_bases{100}{1}(:,p), [800, 6]);
    basis = 10*log10(abs(basis(800/2:800, :)));
    imagesc(1:6, F, basis);
    set(gca, 'YDir', 'normal');
    ylim([0 300]);
end
%% 
figure
for p=1:obj.n_subspace
    subplot(obj.topo_subspace(1),obj.topo_subspace(2),p);
    basis = reshape(obj.bases{1}(:,p), [800, 6]);
    basis = 10*log10(abs(basis(800/2:800, :)));
    imagesc(1:6, F, basis);
    set(gca, 'YDir', 'normal');
    ylim([0 300]);
end

%% 

figure;
train_sample = 10*log10(abs(train_sample_collection(FFTL/2:FFTL,:)));
imagesc(T, F, train_sample);
title('Acoustic Spectrogram');
xlabel('Frames');
ylabel('Frequency (Hz)');
set(gca, 'YDir', 'normal');



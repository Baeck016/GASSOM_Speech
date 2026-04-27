clear all;
clc;

% add dataset
dataset_dir = './dataset/Data_collected/Participant14/Cantonese_sentence/Normal'; % this is a very small dataset %dataset segregated & data not segregated
listing = dir(dataset_dir); %
% get info of dataset
listing = listing(~ismember({listing.name}, {'.', '..'}));
len_listing = length(listing);

save_folder = "./dataset/Data_collected/Participant14/Cantonese_sentence/Normal_mod";

if ~exist(save_folder)
    mkdir(save_folder)
end

%initialize the first audio dataset
targetFs = 16000;   % desired sample rate

for j = 1:len_listing
    dB_gain = db2mag(20);
    [data_sound_1, Fs] = audioread([dataset_dir '/' listing(j).name]);

    % bandpass filter
    begin_freq = 100 / (Fs/2);
    end_freq = 7000 / (Fs/2);
    [b, a] = butter(4, [begin_freq, end_freq], 'bandpass');
    data_sound_1 = filter(b, a, data_sound_1);
    data_sound_1 = data_sound_1 * dB_gain;

    % resample to 16 kHz
    data_sound_1 = resample(data_sound_1, targetFs, Fs);

    % write at 16 kHz
    audiowrite(sprintf('%s/sentence_%d.wav', save_folder, j), ...
               data_sound_1, targetFs, 'BitsPerSample', 16);
end

%% 

dB_gain = db2mag(13);

[data_sound_1, Fs] = audioread([dataset_dir '/' listing(1).name]);
r_1 = rms(data_sound_1, 1);

[data_sound_2, Fs_2] = audioread([dataset_dir '/' listing(2).name]);
r_2 = rms(data_sound_2);

data_sound_1 = data_sound_1 * dB_gain;

begin_freq = 100/ (Fs/2);
end_freq = 7000 / (Fs/2);
[b, a] = butter(4, [begin_freq, end_freq], 'bandpass');
data_sound_1 = filter(b, a, data_sound_1);

% figure
% plot(data_sound_1)

audiowrite('output_truncated.wav', data_sound_1, Fs, 'BitsPerSample', 16);
%% 

files = dir("./dataset/Data_collected/Participant14/Cantonese_sentence/Normal_mod/*.wav");
[data_sound, Fs] = audioread(fullfile(files(1).folder, files(1).name));

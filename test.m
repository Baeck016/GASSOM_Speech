% testing by reconstructing an audio signal
clear all
clc 
% parameter for audio segments
fs=16000; % sampling rate
segment_length=0.075*fs;
jump_length=round(0.075/5*fs);

% parameters for test sample
feature_type='abs_stft'; % 
FFTL=800; % STFT window length
OverlapRate=0.9; % overlap rate

% load GASSOM model
model="./modelset/TIMIT_onesentence/Single_Result/GASSOM_it=10k";
load(model)
% load test audio
    data_nb = "F1";
    file_path = strcat('./dataset/TIMIT_onesentence/test/', data_nb, '.WAV');
    data = audioread(file_path); % reconstruct a training data for testing
    data=data./max(abs(data));
    len_data=length(data);

    %
    test_sample_collection=[];
    test_sample_est_collection=[];
    maxProj_node_track=[];
    proj_node_track = [];
    % 
    max_numsampl=round((len_data-segment_length)/jump_length);
    k=1; % index inside data
    for i=1:max_numsampl
        if k+segment_length-1>len_data % zero-padding
            segment=zeros(segment_length,1);
            segment(1:len_data+1-k)=data(k:len_data);
            return
        else
            segment=data(k:k+segment_length-1);
            k=k+jump_length;
        end
        switch feature_type
            case 'abs_fft'
                test_sample=abs(fft(segment));
                test_sample=test_sample(1:dim_patch_single);
            case 'abs_stft'
                test_sample=abs(stft(segment,'Window',hann(FFTL,'periodic'),...
            'FFTLength',FFTL,'OverlapLength',round(FFTL*OverlapRate)));
                test_sample=test_sample(:);
            case 'mfcc'
                test_sample=mfcc(segment, fs);
                test_sample=test_sample(2:end)';
        end
        % normalize representation
        norm_coeff=norm(test_sample);
        test_sample=test_sample/norm_coeff;
        [winners] = assomEncode(obj,test_sample); % select the winner encode
        
        winners_record(i)=winners;

        % reconstruct by the node with maximal projection
        projection = obj.Proj;
        [~,maxProj_node] = max(projection);
        projection(maxProj_node) = [];
        [~,maxProj_node_2] = max(projection);

        %[~,maxProj_node]=max(obj.Proj);
        proj_node = obj.Proj;
        proj_node_track = [proj_node_track proj_node];
        maxProj_node_track(i) = maxProj_node;
        maxProj_node_track_2(i) = maxProj_node_2;

        test_sample_est= (obj.bases{1}(:,maxProj_node) * obj.coef{1}(maxProj_node) + obj.bases{1}(:,maxProj_node_2) * obj.coef{1}(maxProj_node_2))/2; 
        %+ obj.bases{2}(:,maxProj_node)* obj.coef{2}(maxProj_node); % L X 1
        % 
        test_sample=test_sample*norm_coeff;
        test_sample_est=test_sample_est*norm_coeff;
        % reshape
        test_sample=reshape(test_sample,FFTL,[]);
        test_sample_est=reshape(test_sample_est,FFTL,[]);
        % add phase
        %test_sample_est = test_sample_est.*exp(1j*imag(log(test_sample_est)));
        %
        test_sample_collection=[test_sample_collection test_sample];
        test_sample_est_collection=[test_sample_est_collection test_sample_est];
    end

%%
%basis = abs(obj.bases{1});
%obj.bases{1} = basis;
visualizeBases(obj, [FFTL 6])


%%
figure
plot(data);
axis([0 len_data -1 1])
xlabel('time (second)')
xticks([1 fs 2*fs 3*fs]);
xticklabels({'0', '1','2','3'})
ylabel('amplitude')
title('acoustic waveform')

% plot auditory spectrogram
figure;
subplot(2,1,1);
imagesc(test_sample_collection(1:FFTL/2,:));
title('Acoustic Spectrogram');
xlabel('Time (seconds)');
xticks([1, 201, 401, 601 801]);
xticklabels({'0', '1','2','3', '4'});
ylabel('Frequency (Hz)');
yticks([1 100 200 300 400]);
yticklabels({'8000','6000','4000','2000','0'});

% plot reconstructed auditory spectrogram
subplot(2,1,2);
imagesc(test_sample_est_collection(1:FFTL/2,:));
title('Reconstructed Spectrogram');
xlabel('Time (seconds)');
xticks([1, 201, 401, 601 801]);
xticklabels({'0', '1','2','3', '4'});
ylabel('Frequency (Hz)');
yticks([1 100 200 300 400]);
yticklabels({'8000','6000','4000','2000','0'});


%%
% Calculate the time duration based on the number of samples and the sampling rate
numSamplesSpeech = length(data);
numSamplesNodes = length(maxProj_node_track);
timeDurationNodes = numSamplesNodes / 33.1;

% Generate the time vectors based on the time durations
timeVectorNodes = linspace(0, timeDurationNodes, numSamplesNodes);

% Create a figure for the plots
figure;
   
% Plot the speech amplitude time history
subplot(4,1,1);
plot(data);
axis([0 len_data -1 1])
xlabel('time (second)')
xticks([1 fs 2.02*fs 3.02*fs 4.02*fs]);
xticklabels({'0', '1','2','3', '4'})
ylabel('amplitude')
title('Acoustic Waveform')
grid on;

% Plot the winning node number as a function of time
subplot(4,1,2);
plot(maxProj_node_track);
axis([0 91 0, 100])
xlabel('Time (seconds)');
xticks([1 100/3 100*0.667 100 133]);
xticklabels({'0', '1','2','3', '4'})
ylabel('Winning Node Number');
title('Winning Node Number as a Function of Time');
grid on;

% Plot the acoustic spectrogram
subplot(4,1,3);
imagesc(test_sample_collection(1:FFTL/2,:));
%colorbar;
title('Acoustic Spectrogram');
xlabel('Time (seconds)');
xticks([1, 201, 401, 601 801]);
xticklabels({'0', '1','2','3', '4'});
ylabel('Frequency (Hz)');
yticks([1 100 200 300 400]);
yticklabels({'8000','6000','4000','2000','0'});

% Plot the reconstructed spectrogram
subplot(4,1,4);
imagesc(test_sample_est_collection(1:FFTL/2,:));
%colorbar;
title('Reconstructed Spectrogram');
xlabel('Time (seconds)');
xticks([1, 201, 401, 601 801]);
xticklabels({'0', '1','2','3', '4'});
ylabel('Frequency (Hz)');
yticks([1 100 200 300 400]);
yticklabels({'8000','6000','4000','2000','0'});

% Adjust the spacing between the subplots
spacing = 0.05;
for i = 1:4
    subplot(4,1,i);
    pos = get(gca, 'Position');
    pos(3) = pos(3) - spacing/2;
    set(gca, 'Position', pos);
end


%%
% plot auditory spectrogram
% figure;
% imagesc(test_sample_collection(1:160, 601:612));
% title('Acoustic Spectrogram Segment 101-102');
% xlabel('Time (ms)');
% xticks([0, 6, 12]);
% xticklabels({'0', '30','60'});
% ylabel('Frequency (Hz)');
% yticks([1 40 80 120 160]);
% yticklabels({'8000','6000','4000','2000','0'});

%%
% plot auditory spectrogram
% figure;
% imagesc(test_sample_est_collection(1:160, 601:612));
% title('Reconstructed Spectrogram Segment 101-102');
% xlabel('Time (ms)');
% xticks([0, 6, 12]);
% xticklabels({'0', '30','60'});
% ylabel('Frequency (Hz)');
% yticks([1 40 80 120 160]);
% yticklabels({'8000','6000','4000','2000','0'});


%%
mse = mean((test_sample_collection - test_sample_est_collection).^2, 'all');

%%

% Your existing script...

% Reconstruction
original_audio = data;

reconstructed_audio = istft(test_sample_est_collection, 'Window', hann(FFTL, 'periodic'), ...
    'FFTLength', FFTL, 'OverlapLength', round(FFTL * OverlapRate));
reconstructed_audio = real(reconstructed_audio);
reconstructed_audio = reconstructed_audio ./ max(abs(reconstructed_audio)); % Normalize

% Play the original audio
% disp('Playing original audio...');
% sound(original_audio, fs);
% pause(length(original_audio)/fs + 1); % Wait for the audio to finish playing

% Play the reconstructed audio
disp('Playing reconstructed audio...');
sound(reconstructed_audio, fs);
pause(length(reconstructed_audio)/fs + 1); % Wait for the audio to finish playing


%for i = 1:100
    visualizeNodeSpectrogram(obj, [800 6], 4)
    %saveas(gcf,['./modelset/TIMIT_onesentence/150ms_1600/Basis_Function/' num2str(i) '.jpg']);
%end

function visualizeNodeSpectrogram(this, dim_patch, node_index)
    figure;
    for h = 1:1
        %subplot(2, 1, h);
        basis = this.bases{h}(:, node_index); % Get the basis vectors for node 1
        
        % Reshape the basis vectors to match the specified dimension of the patch
        basis = reshape(basis, dim_patch);

        % Plot the spectrogram for node 
        imagesc(basis(1:400, :));
        title(['Spectrogram for Node ' num2str(node_index)]);
        xlabel('Time(ms)');
        xticks([1, 2, 3, 4, 5, 6])
        xticklabels({'12.5', '37.5', '62.5', '87.5', '112.5', '137.5'})
        ylabel('Frequency');
        yticks([1 200 400 600 800]);
        yticklabels({'8000','6000','4000','2000','0'});
        grid on;
    end
    
end
%%
function visualizeBasis(this,dim_patch)
figure
    for p = 1:100
        subplot(obj.topo_subspace(1), obj.topo_subspace(2),p);
        basis = this(:,p);
        basis = reshape(basis,dim_patch);
        imagesc(basis(1:160, :));
        set(gca,'xtick',[])
        set(gca,'ytick',[])
    end
end

%%
% Reshape the spectrograms into column vectors
% original_vector = test_sample_collection(:);
% reconstructed_vector = test_sample_est_collection(:);

% Calculate the correlation coefficient
% corr_coeff = corrcoef(original_vector, reconstructed_vector);
% corr_value = corr_coeff(1, 2); % Extract the correlation coefficient value

% Display the correlation coefficient
% ACORR(z) = corr_value;

% Calculate MSE
%mse = mean((test_sample_collection - test_sample_est_collection).^2, 'all');

% Calculate SSIM
% ssim_value = ssim(test_sample_collection, test_sample_est_collection);
% ASSIM(z) = ssim_value;

% z = z + 1
% end

% plot acoustic waveform

% reconstruction
clear all
clc
% parameter for audio sample
fs=16000; % sampling rate
stft_sampl_len=6;
FFT_len=800;
Overlap_rate=0.9;

% load GASSOM model
model="./modelset/TIMIT_onesentence/Mix_Result/GASSOM_it=41.983k";
load(model)

%loading test audio datasetLI
directory = './dataset/TIMIT_onesentence/test_SA1';

% result directory
result_dir='modelset/TIMIT_onesentence/Full_SA1_Result/Plot_PCA-Kmeans'; 
if ~exist(result_dir, 'dir')
    mkdir(result_dir)
end

file_pattern_male = fullfile(directory, 'SA1_M*.WAV');
m_audio_files = dir(file_pattern_male);
data1_males = cell(1, length(m_audio_files));
for i = 1:length(m_audio_files)
    data1_males{i} = audioread(fullfile(directory, m_audio_files(i).name));
end

file_pattern_female = fullfile(directory, 'SA1_F*.WAV');
f_audio_files = dir(file_pattern_female);
data1_females = cell(1, length(f_audio_files));
for i = 1:length(f_audio_files)
    data1_females{i} = audioread(fullfile(directory, f_audio_files(i).name));
end

%{
data1_na='M12';
data1=audioread(['./dataset/TIMIT_onesentence/test/' data1_na '.WAV']);
data1=data1./max(abs(data1)); %normalize using peak value
len_data1=length(data1);

data2_na='F12';
data2=audioread(['./dataset/TIMIT_onesentence/test/' data2_na '.WAV']);
data2=data2./max(abs(data2)); %normalize using peak value
len_data2=length(data2);

%matching the length of the data1 and data2 by adding random data with low
%noise by multiplying 1e-3 to the remaining length. 

if len_data1<len_data2
    data1=[data1; 1e-3*rand(len_data2-len_data1,1)];
else
    data2=[data2; 1e-3*rand(len_data1-len_data2,1)];
end
%}
%% 
        % PCA with K-means
        n_bases = abs(obj.bases{1});      % matrix: features x bases

        % Each basis vector (column) becomes one observation (row) for PCA
        X = n_bases';                     % matrix: bases x features
        
        % Standardize data
        X = zscore(X);
        
        % PCA
        [coeff, score, ~, ~, explained, mu] = pca(X);
        % coeff = pca(X);
        
        % Choose number of principal components to keep (e.g., enough for 90% variance)
        cumulativeVariance = cumsum(explained);
        numComponents = find(cumulativeVariance >= 95, 1);
        bases_pca = score(:, 1:numComponents);    % size: bases x numPC
        
        % K-means clustering (reproducible)
        k = 2;                                   % number of clusters
        rng(1);                                  % set random seed for reproducibility
        [idx, C] = kmeans(bases_pca, k, 'Replicates', 10);
        
        % Find indices for each cluster
        find_nodes_1 = find(idx == 1);
        find_nodes_2 = find(idx == 2);
        % find_nodes_3 = find(idx == 3);
        
        % Get basis vectors for each cluster
        cluster_1 = n_bases(:, find_nodes_1);
        cluster_2 = n_bases(:, find_nodes_2);
        % cluster_3 = n_bases(:, find_nodes_3);

         % Frequency vector (for 800 frequency bins)
        F = linspace(1, fs/2, FFT_len/2); % Frequencies from 0 to Nyquist frequency (fs/2)
        
        
        figure;
        for i = 1 : size(cluster_1 , 2)
            subplot(7, 7, i)
            cluster_basis = cluster_1(:, i);
            cluster_basis = reshape(cluster_basis, [800, 6]);
            cluster_basis = 10*log10(abs(cluster_basis(FFT_len/2:FFT_len, :)));
            imagesc(1:6, F, cluster_basis);
            set(gca, 'YDir', 'normal');
            ylabel("Frequency (Hz)");
        end

        figure;
        for i = 1 : size(cluster_2 , 2)
            subplot(5, 5, i)
            cluster_basis = cluster_2(:, i);
            cluster_basis = reshape(cluster_basis, [800, 6]);
            cluster_basis = 10*log10(abs(cluster_basis(FFT_len/2:FFT_len, :)));
            imagesc(1:6, F, cluster_basis);
            set(gca, 'YDir', 'normal');
            ylabel("Frequency (Hz)");
        end
        
        figure;
        for i = 1 : size(cluster_3 , 2)
            subplot(6, 6, i)
            cluster_basis = cluster_3(:, i);
            cluster_basis = reshape(cluster_basis, [800, 6]);
            cluster_basis = 10*log10(abs(cluster_basis(FFT_len/2:FFT_len, :)));
            imagesc(1:6, F, cluster_basis);
            set(gca, 'YDir', 'normal');
            ylabel("Frequency (Hz)");
        end
        
%%

mse_collection = [];
winning_node_m = [];
winning_node_f = [];
index = 1;

for males_index = 1 : length(m_audio_files)
    for females_index = 1 : length(f_audio_files)

        m_char = m_audio_files(males_index).name;
        f_char = f_audio_files(females_index).name;

        m_char = split(m_char, '.');
        m_char = m_char{1};
        
        f_char = split(f_char, '.');
        f_char = f_char{1};
        combined_char = [m_char, '_' , f_char];

        % result directory
        result_plot_dir = [result_dir, '/', combined_char];
        
        if ~exist(result_plot_dir, 'dir')
            mkdir(result_plot_dir)
        end
        %% 


        %matching the length of data1 and data2 based on the shorter length by
        %truncating
        data1 = data1_males{males_index};
        data1 = data1./max(abs(data1));
        len_data1 = length(data1);
        
        data2 = data1_females{females_index};
        data2 = data2./max(abs(data2));
        len_data2 = length(data2);
        
        if len_data1<len_data2
            data2 = data2(1:len_data1);
        else
            data1 = data1(1:len_data2);
        end
        
        %mixing data1 and data2, and peak normalizing
        mix_data=data1+data2;
        mix_data=mix_data./max(abs(mix_data));
        
        % stft
        DATA1=stft(data1,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        DATA2=stft(data2,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        MIX_DATA=stft(mix_data,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        
        len_data=size(MIX_DATA,2);
        max_numsampl=floor(len_data/stft_sampl_len);
        
        Mix_collection=[]; % for mixed acoustic spectrogram (c) and (d)
        mix_collection=[];
        
        Recon_allnode_collection=[]; % for reconstruction by all nodes
        
        SP1_collection=[]; % for acoustic spectrogram of SP1 (a)
        SP1_est_collection=[]; % for reconstructed spectrogram of single SP1 (e)
        Attend_SP1_collection=[]; %  (g) for reconstructed spectrogram of Mix: attedn SP1 (g)
        SP1_node_track=[]; % record the most popular node for SP1 before mixing
        
        SP2_collection=[]; % for acoustic spectrogram of SP2 (b)
        SP2_est_collection=[]; % for reconstructed spectrogram of single SP2 (f)
        Attend_SP2_collection=[]; % for reconstructed spectrogram of Mix: attedn SP2 (h)
        SP2_node_track=[]; % record the most popular node for SP2 before mixing        
        
        %% Reconstruction
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%                              %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%% Single Reconstruction of SP1 %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%                              %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        k=1;
        for i=1:max_numsampl
            SP1=DATA1(:,k:k+stft_sampl_len-1);
            k=k+stft_sampl_len;
            SP1=SP1(:);
            % normalize representation
            norm_coeff=norm(abs(SP1));
            SP1=SP1/norm_coeff;
            [winners] = assomEncode(obj,abs(SP1)); % calculate Projection by assomEncode method
        
            % reconstruct by the node with maximal projection
            [~,maxProj_node]=max(obj.Proj);
            SP1_node_track(i)=maxProj_node;
            SP1_est=obj.bases{1}(:,maxProj_node) * obj.coef{1}(maxProj_node); 
            %+obj.bases{2}(:,maxProj_node)* obj.coef{2}(maxProj_node); % L X 1
            
            % 
            SP1=SP1*norm_coeff;
            SP1_est=SP1_est*norm_coeff;
            % reshape
            SP1=reshape(SP1,FFT_len,[]);
            SP1_est=reshape(SP1_est,FFT_len,[]);
            % add phase
            SP1_est=SP1_est.*exp(1j*imag(log(SP1)));
            %
            SP1_collection=[SP1_collection SP1];
            SP1_est_collection=[SP1_est_collection SP1_est];
        end
        sp1=istft(SP1_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        single_sp1=istft(SP1_est_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        single_sp1=real(single_sp1);
        single_sp1=single_sp1./max(abs(single_sp1));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%                              %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%% Single Reconstruction of SP2 %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%                              %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        k=1;
        for i=1:max_numsampl
            SP2=DATA2(:,k:k+stft_sampl_len-1);
            k=k+stft_sampl_len;
            SP2=SP2(:);
            % normalize representation
            norm_coeff=norm(abs(SP2));
            SP2=SP2/norm_coeff;
            [winners] = assomEncode(obj,abs(SP2)); % calculate Projection by assomEncode method
            % reconstruct by the node with maximal projection
            [~,maxProj_node]=max(obj.Proj);
            maxProj_node = winners;
            SP2_node_track(i)=maxProj_node;
            SP2_est=obj.bases{1}(:,maxProj_node) * obj.coef{1}(maxProj_node);
            %+obj.bases{2}(:,maxProj_node)* obj.coef{2}(maxProj_node); % L X 1
            
            % 
            SP2=SP2*norm_coeff;
            SP2_est=SP2_est*norm_coeff;
            % reshape
            SP2=reshape(SP2,FFT_len,[]);
            SP2_est=reshape(SP2_est,FFT_len,[]);
            % add phase
            SP2_est=SP2_est.*exp(1j*imag(log(SP2)));
            %
            SP2_collection=[SP2_collection SP2];
            SP2_est_collection=[SP2_est_collection SP2_est];
        end
        sp2=istft(SP2_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        single_sp2=istft(SP2_est_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        single_sp2=real(single_sp2);
        single_sp2=single_sp2./max(abs(single_sp2));
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%                                    %%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% Mix attend Reconstruction of SP1/2 %%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%                                    %%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 
        k=1; % index inside data
        for i=1:max_numsampl
            Mix=MIX_DATA(:,k:k+stft_sampl_len-1);
            k=k+stft_sampl_len;
            Mix=Mix(:);
            % normalize representation
            norm_coeff=norm(abs(Mix));
            Mix=Mix/norm_coeff;
            % 
            [winners] = assomEncode(obj,abs(Mix)); % calculate Projection by assomEncode method
        
            % tmp=obj.Proj(1:5);
            % SP1_node=find(obj.Proj==max(tmp));
            % tmp=obj.Proj(11:15);
            % SP2_node=find(obj.Proj==max(tmp));
        
        
        %Clustering
            % if mean(obj.Proj(1:56))>mean(obj.Proj(57:100)) % Clustering
            %     tmp=obj.Proj(1:56);
            %     SP1_node=find(obj.Proj==max(tmp));
            %     tmp=rmoutliers(obj.Proj(57:100),'percentiles',[0 90]); % remove outlier
            %     SP2_node=find(obj.Proj==max(tmp));
            % else
            %     tmp=rmoutliers(obj.Proj(1:56),'percentiles',[0 60]); % remove outlier
            %     SP1_node=find(obj.Proj==max(tmp));
            %     tmp=obj.Proj(57:100);
            %     SP2_node=find(obj.Proj==max(tmp));
            % end
        
        %Clustering with K-mean
            % 
            % basis = obj.bases{1};
            % % features = mean(basis, 1);
            % % features = (features - min(features)) / (max(features) - min(features));
            % [idx,centroids] = kmeans(basis',2);
            % 
            % cluster1_values = obj.Proj(idx == 1);
            % [max_value1, relative_index1] = max(cluster1_values);
            % cluster1_nodes = find(idx == 1);
            % original_index1 = cluster1_nodes(relative_index1);
            % 
            % 
            % cluster2_values = obj.Proj(idx == 2);
            % [max_value2, relative_index2] = max(cluster2_values);
            % cluster2_nodes = find(idx == 2);
            % original_index2 = cluster2_nodes(relative_index2);
            % 
            % if cluster1_values > cluster2_values
            %     [idx_sub_1, centroids_sub] = kmeans(basis(:, cluster1_nodes)',2);
            % 
            %     subcluster1_values = cluster1_values(idx_sub_1 == 1);
            %     subcluster2_values = cluster1_values(idx_sub_1 == 2);
            % 
            %     [max_value_sub1, relative_index_sub1] = max(subcluster1_values);
            %     [max_value_sub2, relative_index_sub2] = max(subcluster2_values);
            % 
            %     subcluster1_nodes = cluster1_nodes(idx_sub_1 == 1);
            %     subcluster2_nodes = cluster1_nodes(idx_sub_1 == 2);
            % 
            %     original_index_sub1 = subcluster1_nodes(relative_index_sub1);
            %     original_index_sub2 = subcluster2_nodes(relative_index_sub2);
            % 
            %     if max_value_sub1 > max_value_sub2
            %         SP1_node = original_index_sub1;
            %         SP2_node = original_index_sub2;
            % 
            %     else
            %         SP1_node = original_index_sub2;
            %         SP2_node = original_index_sub1;
            %     end
            % else
            %     [idx_sub_2, centroids_sub_2] = kmeans(basis(:,cluster2_nodes)',2);
            % 
            %     subcluster1_values = cluster2_values(idx_sub_2 == 1);
            %     subcluster2_values = cluster2_values(idx_sub_2 == 2);
            % 
            %     [max_value_sub1, relative_index_sub1] = max(subcluster1_values);
            %     [max_value_sub2, relative_index_sub2] = max(subcluster2_values);
            % 
            %     subcluster1_nodes = cluster2_nodes(idx_sub_2 == 1);
            %     subcluster2_nodes = cluster2_nodes(idx_sub_2 == 2);
            % 
            %     original_index_sub1 = subcluster1_nodes(relative_index_sub1);
            %     original_index_sub2 = subcluster2_nodes(relative_index_sub2);
            % 
            %     if max_value_sub1 > max_value_sub2
            %         SP1_node = original_index_sub1;
            %         SP2_node = original_index_sub2;
            % 
            %     else
            %         SP1_node = original_index_sub2;
            %         SP2_node = original_index_sub1;
            %     end
            % end
        
            % Any means with K-means
            n_projected = obj.Proj; 

            [cluster_2_values, relative_index_2] = max(n_projected(find_nodes_2));
            SP1_node = find_nodes_2(relative_index_2);

            [cluster_3_values, relative_index_3] = max(n_projected(find_nodes_3));
            SP2_node = find_nodes_3(relative_index_3);
            % 
            SP1_node_track_est(i)=SP1_node;
            SP2_node_track_est(i)=SP2_node;
            
            % reconstructed by the specified nodewe
            Attend_SP1=obj.bases{1}(:,SP1_node) * obj.coef{1}(SP1_node);
            %+obj.bases{2}(:,SP1_node)* obj.coef{2}(SP1_node); % L X 1
            
            Attend_SP2=obj.bases{1}(:,SP2_node) * obj.coef{1}(SP2_node);
            %+obj.bases{2}(:,SP2_node)* obj.coef{2}(SP2_node); % L X 1
            
            % reconstructed by all node
            Recon_allnode=obj.bases{1}(:,1) * obj.coef{1}(1);
            %+obj.bases{2}(:,1)* obj.coef{2}(1);
            for ind_node=1:100
                Recon_allnode=Recon_allnode+obj.bases{1}(:,ind_node) * obj.coef{1}(ind_node);
                %+obj.bases{2}(:,ind_node)* obj.coef{2}(ind_node);
            end
            Recon_allnode=Recon_allnode/100;
            % 
            Mix=Mix*norm_coeff;
            Attend_SP1=Attend_SP1*norm_coeff;
            Attend_SP2=Attend_SP2*norm_coeff;
            Recon_allnode=Recon_allnode*norm_coeff;
            % reshape
            Mix=reshape(Mix,FFT_len,[]);
            Attend_SP1=reshape(Attend_SP1,FFT_len,[]);
            Attend_SP2=reshape(Attend_SP2,FFT_len,[]);
            Recon_allnode=reshape(Recon_allnode,FFT_len,[]);
            % add phase
            Attend_SP1=Attend_SP1.*exp(1j*imag(log(Mix)));
            Attend_SP2=Attend_SP2.*exp(1j*imag(log(Mix)));
            %
            Mix_collection=[Mix_collection Mix];
            Attend_SP1_collection=[Attend_SP1_collection Attend_SP1];
            Attend_SP2_collection=[Attend_SP2_collection Attend_SP2];
            Recon_allnode_collection=[Recon_allnode_collection Recon_allnode];

        end
        mix=istft(Mix_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        attend_sp1=istft(Attend_SP1_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        attend_sp2=istft(Attend_SP2_collection,'Window',hann(FFT_len,'periodic'),...
        'FFTLength',FFT_len,'OverlapLength',round(FFT_len*Overlap_rate));
        attend_sp1=real(attend_sp1);
        attend_sp1=attend_sp1./max(abs(attend_sp1));
        attend_sp2=real(attend_sp2);
        attend_sp2=attend_sp2./max(abs(attend_sp2));

        winning_node_f = [winning_node_f SP1_node_track_est];
        winning_node_m = [winning_node_m SP2_node_track_est];

        % If you want calculate MSE
        mse_sp2_men = mean((abs(SP2_est_collection(FFT_len/2:FFT_len,:)) - abs(Attend_SP2_collection(FFT_len/2:FFT_len,:))).^2, 'all');
        mse_sp2_female = mean((abs(SP1_est_collection(FFT_len/2:FFT_len,:)) - abs(Attend_SP1_collection(FFT_len/2:FFT_len,:))).^2, 'all');
        mse_collection_cluster2_men{index}= {combined_char, mse_sp2_men, mse_sp2_female};

        mse_sp1_men = mean((abs(SP1_est_collection(FFT_len/2:FFT_len,:)) - abs(Attend_SP2_collection(FFT_len/2:FFT_len,:))).^2, 'all');
        mse_sp1_female = mean((abs(SP2_est_collection(FFT_len/2:FFT_len,:)) - abs(Attend_SP1_collection(FFT_len/2:FFT_len,:))).^2, 'all');
        mse_collection_cluster1_men{index} = {combined_char, mse_sp1_men, mse_sp1_female};

        index = index + 1;
        
        %%
        % Create a figure for the plots
        %{
        figure;
            
        % Plot the speech amplitude time history
        subplot(3,1,1);
        plot(mix_data);
        axis([0 length(mix_data) -1 1])
        xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs 4*fs]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('amplitude')
        title('Acoustic aveform of Mix')
        grid on;
        
        % Plot the acoustic spectrogram
        subplot(3,1,2);
        imagesc(abs(Mix_collection(1:FFT_len/2,:)));
        title('Acoustic Spectrogram of Mix')
        xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        % Plot the reconstructed spectrogram
        subplot(3,1,3);
        imagesc((abs(Recon_allnode_collection(1:FFT_len/2,:))));
        %colorbar;
        title('Reconstructed Spectrogram of Mix');
        xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        % Adjust the spacing between the subplots
        spacing = 0.05;
        for i = 1:3
            subplot(3,1,i);
            pos = get(gca, 'Position');
            pos(3) = pos(3) - spacing/2;
            set(gca, 'Position', pos);
        end
        
        %%
        % Create a figure for the plots female (non-mix)
        figure;
        
        %Waveform
        subplot(3,1,1)
        plot(data1);
        axis([0 length(data2) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Acoustic Waveform of SP1')   
        
        % Plot the speech amplitude time history
        subplot(3,1,2);
        imagesc(abs(SP1_collection(1:FFT_len/2,:)));
        title('Acoustic Spectrogram of SP1')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        grid on;
        
        % Plot the acoustic spectrogram
        subplot(3,1,3);
        imagesc((abs(SP1_est_collection(1:FFT_len/2,:))));
        title('Reconstructed Acoustic Spectrogram of SP1')
        xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        % Adjust the spacing between the subplots
        spacing = 0.05;
        for i = 1:3
            subplot(3,1,i);
            pos = get(gca, 'Position');
            pos(3) = pos(3) - spacing/2;
            set(gca, 'Position', pos);
        end
        %}
        
        %%
        %{
        % Create a figure for the plots Male (non-mix)
        figure;
        
        %Waveform
        subplot(3,1,1)
        plot(data2);
        axis([0 length(data2) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Acoustic Waveform of SP2')   
        
        % Plot the speech amplitude time history
        subplot(3,1,2);
        imagesc(abs(SP2_collection(1:FFT_len/2,:)));
        title('Acoustic Spectrogram of SP2')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        grid on;
        
        % Plot the acoustic spectrogram
        subplot(3,1,3);
        imagesc((abs(SP2_est_collection(1:FFT_len/2,:))));
        title('Reconstructed Acoustic Spectrogram of SP2')
        xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        
        % Adjust the spacing between the subplots
        spacing = 0.05;
        for i = 1:3
            subplot(3,1,i);
            pos = get(gca, 'Position');
            pos(3) = pos(3) - spacing/2;
            set(gca, 'Position', pos);
        end
        %%
        %}

        % Frequency vector (for 800 frequency bins)
        F = linspace(1, fs/2, FFT_len/2); % Frequencies from 0 to Nyquist frequency (fs/2)
        
        % Time vector (time frames)
        T = linspace(1, size(SP1_collection,2), size(SP1_collection,2)); % Adjust based on your segment duration
        
        % Create a figure for the plots Female (SP1 MIX)
        fig = figure('Visible','off');
        %figure;
        subplot(7,1,1)
        plot(data1);
        axis([0 length(data2) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Female Acoustic Waveform')
        
        % Plot the speech amplitude time history
        subplot(7,1,2);
        SP1_collection = 10*log10(abs(SP1_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, SP1_collection);
        title('Female Acoustic Spectrogram')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        set(gca, 'YDir', 'normal');
        grid on;
        
        % Mix waveform
        subplot(7,1,3)
        plot(mix_data);
        axis([0 length(mix_data) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Acoustic Waveform of Mix')
        
        %Acoustic Spectrogram of Mix
        subplot(7,1,4)
        Mix_collection = 10*log10(abs(Mix_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, Mix_collection);
        title('Acoustic spectrogram of Mix')
        % xlabel('time (second)')
        ylabel('Frequency (kHz)')
        set(gca, 'YDir', 'normal');

        % Plot the acoustic spectrogram
        subplot(7,1,5);
        SP1_est_collection = 10*log10(abs(SP1_est_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, SP1_est_collection);
        title('Female: Ground Truth Reconstructed Spectrogram')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        set(gca, 'YDir', 'normal');

        % Plot the reconstructed spectrogram
        subplot(7,1,6);
        Attend_SP1_collection = 10*log10(abs(Attend_SP1_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, Attend_SP1_collection);
        title('Female: Reconstructed Spectrogram from Mix')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        set(gca, 'YDir', 'normal');

        subplot(7,1,7);
        plot(SP1_node_track_est);
        axis([0 94 0, 100])
        % xlabel('Time (seconds)');
        % xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        disp(SP1_node_track_est)
        
        spacing = 0.03;  % Increase this for more space between subplots
        n = 7;           % Number of subplots
        
        for i = 1:n
            subplot(n,1,i);
            pos = get(gca, 'Position');
            
            % Reduce height to make room for spacing
            pos(4) = pos(4) - spacing;
            
            set(gca, 'Position', pos);
        end

        save_path = fullfile(result_plot_dir, 'female_spectrogram.jpg');

        % Save the figure
        saveas(gcf, save_path);

        
        %%

        % Frequency vector (for 800 frequency bins)
        F = linspace(1, fs/2, FFT_len/2); % Frequencies from 0 to Nyquist frequency (fs/2)
        
        % Time vector (time frames)
        T = linspace(1, size(SP2_collection,2), size(SP2_collection,2)); % Adjust based on your segment duration
        
        % Create a figure for the plots Male (SP2 MIX)
        fig = figure('Visible','off');
        %figure;
        subplot(7,1,1)
        plot(data2);
        axis([0 length(data2) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Male Acoustic Waveform')
        
        % Plot the speech amplitude time history
        subplot(7,1,2);
        SP2_collection = 10*log10(abs(SP2_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, SP2_collection);
        title('Male Acoustic Spectrogram')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        set(gca, 'YDir', 'normal');
        grid on;
        
        % Mix waveform
        subplot(7,1,3)
        plot(mix_data);
        axis([0 length(mix_data) -1 1])
        %xlabel('time (second)')
        xticks([1 fs 2*fs 3*fs]);
        xticklabels({'0', '1','2','3'})
        ylabel('amplitude')
        title('Acoustic Waveform of Mix')
        
        %Acoustic Spectrogram of Mix
        subplot(7,1,4)
        imagesc(T, F, Mix_collection);
        title('Acoustic spectrogram of Mix')
        % xlabel('time (second)')
        ylabel('Frequency (kHz)')
        set(gca, 'YDir', 'normal');
        
        % Plot the acoustic spectrogram
        subplot(7,1,5);
        SP2_est_collection = 10*log10(abs(SP2_est_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, SP2_est_collection);
        title('Male: Ground Truth Reconstructed Spectrogram')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        set(gca, 'YDir', 'normal');
        
        % Plot the reconstructed spectrogram
        subplot(7,1,6);
        Attend_SP2_collection = 10*log10(abs(Attend_SP2_collection(FFT_len/2:FFT_len,:)));
        imagesc(T, F, Attend_SP2_collection);
        title('Male: Reconstructed Spectrogram from Mix')
        % xlabel('Time (seconds)');
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        set(gca, 'YDir', 'normal');
        
        % Plot the reconstructed spectrogram
        subplot(7,1,7);
        plot(SP2_node_track_est);
        axis([0 94 0, 100])
        % xlabel('Time (seconds)');
        xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        disp(SP2_node_track_est)
        
        % Adjust the spacing between the subplots
        spacing = 0.03;  % Increase this for more space between subplots
        n = 7;           % Number of subplots
        
        for i = 1:n
            subplot(n,1,i);
            pos = get(gca, 'Position');
            
            % Reduce height to make room for spacing
            pos(4) = pos(4) - spacing;
            
            set(gca, 'Position', pos);
        end

        save_path = fullfile(result_plot_dir, 'male_spectrogram.jpg');

        % Save the figure
        saveas(gcf, save_path);
        
        %% 
        
        %{
        %% SP1 Node Comparison
        figure;
        
        subplot(4,1,1);
        imagesc(abs(SP1_est_collection(1:FFT_len/2,:)));
        title('Female: Ground Truth Reconstructed Spectrogram')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        subplot(4,1,2);
        plot(SP1_node_track);
        axis([0 94 0, 100])
        xlabel('Time (seconds)');
        xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        
        subplot(4,1,3);
        imagesc((abs(Attend_SP1_collection(1:FFT_len/2,:))));
        title('Female: Reconstructed Spectrogram from Mixed')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        subplot(4,1,4);
        plot(SP1_node_track_est);
        axis([0 94 0, 100])
        xlabel('Time (seconds)');
        xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        
        %% SP2 Node Comparison
        
        figure;
        
        subplot(4,1,1);
        imagesc(abs(SP2_est_collection(1:FFT_len/2,:)));
        title('Male: Ground Truth Reconstructed Spectrogram')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        subplot(4,1,2);
        plot(SP2_node_track);
        axis([0 94 0, 100])
        xlabel('Time (seconds)');
        xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        
        subplot(4,1,3);
        imagesc((abs(Attend_SP2_collection(1:FFT_len/2,:))));
        title('Male: Reconstructed Spectrogram from Mixed')
        %xlabel('Time (seconds)');
        xticks([1, 197, 2*197, 3*197 4*197]);
        xticklabels({'0', '1','2','3', '4'});
        ylabel('Frequency (Hz)');
        yticks([1 100 200 300 400]);
        yticklabels({'8000','6000','4000','2000','0'});
        
        subplot(4,1,4);
        plot(SP2_node_track_est);
        axis([0 94 0, 100])
        xlabel('Time (seconds)');
        xticks([1 32.8 2*32.8 3*32.8 4*32.8]);
        xticklabels({'0', '1','2','3', '4'})
        ylabel('Winning Node Number');
        title('Winning Node Number as a Function of Time');
        grid on;
        disp(SP2_node_track_est)

        %}

    end
end
%% 

% mse_std = std(mse_collection);
% mse_mean = mean(mse_collection);

% Suppose N is the total number of entries
N = numel(mse_collection_cluster2_men);

% Preallocate arrays
mse_sp2_men_vals = zeros(N,1);
mse_sp2_female_vals = zeros(N,1);

% Extract values from cell array
for i = 1:N
    mse_sp2_men_vals(i) = mse_collection_cluster2_men{i}{2};
    mse_sp2_female_vals(i) = mse_collection_cluster2_men{i}{3};
end

% Calculate mean and std
mean_men = mean(mse_sp2_men_vals);
std_men = std(mse_sp2_men_vals);

mean_female = mean(mse_sp2_female_vals);
std_female = std(mse_sp2_female_vals);

mean_vals_sp2 = [mean_men, mean_female];
std_vals_sp2 = [std_men, std_female];

categories = {'Men', 'Female'};

figure;

% Plot the bar chart
b = bar(mean_vals_sp2);

hold on;

% Add error bars
% X positions of bars
x = 1:length(mean_vals_sp2);
errorbar(x, mean_vals_sp2, std_vals_sp2, '.k', 'LineWidth', 1.5);

% Set category labels on x-axis
set(gca, 'XTick', x, 'XTickLabel', categories);

ylabel('Mean MSE Value');
title('MSE with Men = SP2 and Female = SP1 Cluster');

hold off;
grid on;



%% 
% Suppose N is the total number of entries
N = numel(mse_collection_cluster1_men);

% Preallocate arrays
mse_sp1_men_vals = zeros(N,1);
mse_sp1_female_vals = zeros(N,1);

% Extract values from cell array
for i = 1:N
    mse_sp1_men_vals(i) = mse_collection_cluster1_men{i}{2};
    mse_sp1_female_vals(i) = mse_collection_cluster1_men{i}{3};
end

% Calculate mean and std
mean_men_sp1 = mean(mse_sp1_men_vals);
std_men_sp1 = std(mse_sp1_men_vals);

mean_female_sp1 = mean(mse_sp1_female_vals);
std_female_sp1 = std(mse_sp1_female_vals);

mean_vals_sp1 = [mean_men_sp1, mean_female_sp1];
std_vals_sp1 = [std_men_sp1, std_female_sp1];

figure;

% Plot the bar chart
b = bar(mean_vals_sp1);

hold on;

% Add error bars
% X positions of bars
x = 1:length(mean_vals_sp1);
errorbar(x, mean_vals_sp1, std_vals_sp1, '.k', 'LineWidth', 1.5);

% Set category labels on x-axis
set(gca, 'XTick', x, 'XTickLabel', categories);

ylabel('Mean MSE Value');
title('MSE with Men = SP1 and Female = SP2 Cluster');

hold off;
grid on;

%{
figure;
bar(mse_collection, "stacked"); 
hold on
errorbar(1:length(mse_collection), mse_collection, mse_stds, 'k.', 'LineWidth', 2); % std as error bars

ylabel('Mean Squared Error (Energy Spectrogram)');
title('Speech Separation MSE with Std Deviation');
grid on;
hold off;
%}

%save([result_dir '/GASSOM_MSE.mat'], 'mse_collection')

%%

%[minValue, minIndex] = min(mse_collection);
%[maxValue, maxIndex] = max(mse_collection);

%%
% plot histgram and heatmap of winning node
Node_updateCount=zeros(100,1);
for i=1:100
    Node_updateCount(i)=sum(winning_node_m==i);
end
figure
bar(Node_updateCount);
xlabel('node index');
saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/hist_updated_node_male.jpg')
Node_updateCount_topo=reshape(Node_updateCount,10,10);
Node_updateCount_topo=Node_updateCount_topo'; 
figure
heatmap(Node_updateCount_topo);
saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/heatmap_updated_node_male.jpg')

%%
% plot histgram and heatmap of winning node
Node_updateCount=zeros(100,1);
for i=1:100
    Node_updateCount(i)=sum(winning_node_f==i);
end
figure
bar(Node_updateCount);
xlabel('node index');
%saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/hist_updated_node_female.jpg')
Node_updateCount_topo=reshape(Node_updateCount,10,10);
Node_updateCount_topo=Node_updateCount_topo'; 
figure
heatmap(Node_updateCount_topo);
%saveas(gcf,'./modelset/TIMIT_onesentence/1_Bases/heatmap_updated_node_female.jpg')
%% 
% correlation
% [rho,pval] = corr(freq_max_f1', freq_max_f2');










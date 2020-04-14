%% Ben's Complete ERP Analysis Function
% Thanks to:
% - Keisuke Fukuda, University of Toronto Mississauga, 2020.
% - Scott Makeig, Arnaud Delorme, Clemens Brunner SCCN/INC/UCSD, La Jolla,
% 1997.

%% Before Running:
% [] Note, this is designed for use with Brainvision EEG equipment data, 
%    but step 2 (artifact detection) and onward should compatible with any serial data.
% [] Carefully edit the settings according to your experiment and desired
%    output.
% [] EEG/Behavioural Data needed in working directory:
%    '##.eeg'
%    '##.vhdr'
%    '##.vmrk'
%    'experiment_behaviour_data_##.mat'
% [] Functions needed in working directory: 
%    'eegfilt.m'
%    'bva_readmarker.m'
%    'bva_readheader.m'
%    'bva_loadeeg.m'

clear all;
close all;

%% Table of Contents
% (-1) Flag meanings, just for reference.
% (0) Settings.
% --- Subject analyses ---
% (1) Convert Matlab; Convert EEG and Behavioural files to Matlab.
% (2) Artifact Detection; Separate trials with artifacts from clean trials.
% (3) Modular; Do_exp_EEG.
% (4) Merge; ERP and Behavioural data.
% (5) Hilbert: Output amplitude & phase.
% (6) Phase-locking value.
% --- Grand analyses ---
% (7) Do Grand; Create grand EEG file by IV's.
% (8) Plot ERP Grand; Plot ERP's by Time windows.
% (9) Draw Electrode Surface Map Grand.

%% (-1) Flag Meanings
%% RESET FLAGS
% Spacebar pressed: block+210. Blocks 15.
% Real trial start: t. Trials 48.
% Cue Onset: 100+current_blocktype*10+current_lr. Blocktypes 1-3.
% current_lr 1-2.
% Track Onset: 200+shape/color speed (-2, -1, 1, 2).
% Reset Onset: 99.
% Retention Onset: 199.
% Response Onset: 240.
% Trial End: Response Correct (251), Incorrect (250).
%% PACMAN FLAGS
%block+210
%trial
%100+current_target_loc (1/2 LEFT, 3/4 RIGHT) % Present Cue
%100+10*current_pacman+current_direction (start tracking) % Start Track
%%98 Stimulus(middle of two dots) crosses over
%99 Target dot crosses over % Cross Over
%200+10*target_clockwise % Report Screen
%251 Correct
%252 Incorrect
%255 Trial end(resp made)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% (0) SETTINGS
disp('---------------------------------------------------------');
disp('--------------------------BEGIN--------------------------');
disp('---------------------------------------------------------');
disp('(0) Setting up');

%% OS and Basic Settings
os = 1; % Operating system (1 Mac, 2 Windows)

%% Which Analyses to Run?
run_analyses = 2; % 1 Sub only, 2 Grand only, 3 Both. 0 None.
run_hilbert = 1; % 1 Run amplitude/phase analysis (via trial by trial hilbert transform). 0 None.
run_plv = 1; % 1 Run PLV analysis (trial by trial phase-locking-value). 0 None.
run_plot = 2; % 1 Plot ERP, 2 Draw Surfacemap. 0 None.
save_raw_data_files = 0; % 1 Save eeg/erp data. 0 None.

%% Settings for (1) Convert Matlab
% Experiment Segmentation
numblocks = 15;
numtrialsperblock = 48;

% EEG data broken or not; several EEG subject files?
sub_data_broken = 1;
% If EEG data is broken, please fill out this information.
% Subject's number as in {'07'};
sub_num = {'02'};
% Subject's eeg numbers in order; {'07','07.1','07.2',etc.};
sub_eeg_num = {'02','02.1'};

% Subject's behavioural files in order; {'wm_reset_data_15', 'wm_reset_data_15.1', etc.};
sub_beh_prename = {'wm_reset_data_'};

% Subject's behavioural file trial range in each file.
sub_beh_trials = {1,720};

% Behavioural accuracy/RT data.
beh_accrt_conds = {'current_resp_acc','current_resp_rt'};
beh_accrt_nums = [1,2]; % correct/incorrect accuracy. OR if binomial; all x-values.

% Declare all channels. VEOG, LHEOG, and RHEOG are mandatory.
channels = {'VEOG';'AF3';'F3';'Fz';... %
                'L_HEOG';'FC5';'FC1';'C3';...
                'T7';'L_Mastoid';'P7';'P3';...
                'Pz';'POz';'PO3';'PO7';...
                'O1';'O2';'PO4';'PO8';...
                'P4';'P8';'T8';...%R_Mastoid (Ref) Omitted.
                'C4';'Cz';'FC2';'FC6';...
                'R_HEOG';'F4';'AF4';'AFz';...
                };
fs = 250; % Data sampling rate; only needed if running just grand.

%% Settings for (2) Artifact Detection
% Set filter thresholds
blink_threshold = 75; %in microvolts default 75
eMove_threshold = 30; %in microvolts default 30
blocking_threshold = 0.1; %in microvolts 1 is recommended default 0.1
window_size = 50; %number of samples; 50 = 200ms, etc. default 50
step_size = 25; %in samples, gap between each time window. default 25

%% Settings for (3) Modular; Do Reset EEG
% Conditions and their time windows (in data samples; these are the X-AXES)
conditions = {'Cue';'Track';'Reset';'Retention'}; % Name conditions of interest. Flags must be in temporal order (t1,t2,t3,'Reset', etc.)
cond_timewindows = [50,249; 125,249; 50,124; 125,249]; % Timewindows [-n1,+n1; -n2,+n2] from flag, for each condition in #datasamples [-#samples1,+samples1; -samples2,+samples2; 50,249; etc.]
buffer_size = 300; % To give ample time for filters

% Baselining Brainwaves (amplitude)
amplock_cond = 2;% Which condition to timelock to? [1] Cue or 2 Track or 3 Reset, etc.
amplock_timewindow = [50, 0];% What timewindow [-n,+n] from flag, for eegdata in amplock_cond? in #datasamples! e.g., [init, end] -200 to 0ms.

% Event codes (ecs) for each timewindow of interest (condition)
% Important: Flags must be in order; (1)blocks ... (>1)trials ...
block_ecs = [211 225]; % Flag ranges for block ecs
trial_ecs = [1 48]; %Flag ranges for trial ecs

% Flag ranges for each condition [#init1,#end1; #init2,#end2; etc.]
% Important: Flags must be in order; (1)cond1flags, (2)cond2flags, (5)cond3flags, etc.
cond_ecs = [111,132; 198,202; 99,99; 199,199];
final_ecs = [240]; % Flag that comes after the last condition's flag (Response Onset)

% Filtering
range_threshold = 75;
noise_criteria = 120;

%% Settings for (4) Merge; ERP and Behavioural data
% This section includes organizing all data into left=ipsi, right=contra.
% Name which behavioural conditions to merge with ERP; [char('curr_block','curr_lr',etc.)]
beh_merge_conds = [char('current_blocktype','current_lr','current_track_dur', ...
                    'current_reset','current_reset_type','current_reset_time', ...
                    'current_trackover_time','current_diff_probe_sim','current_probe_cw', ...
                    'current_resp_key','current_resp_acc','current_resp_rt')];

% Organize channels by left/right=ipsi/contra index, as in channel(index). 
% If unilateral (like VEOG), just put the same index for both;
% Example: index_l = [1 2 3 4], index_r = [1 30 29 4] etc.
channel_index = [1 2 3 4 ... 
                 5 6 7 8 ...
                 9 10 11 12 ...
                 13 14 15 16 ...
                 17 18 19 20 ...
                 21 22 23 ...
                 24 25 26 27 ...
                 28 29 30 31 ...
                 ];

channel_index_r = [1 30 29 4 ... 
                 5 27 26 24 ...
                 23 10 22 21 ...
                 13 14 19 20 ...
                 18 17 15 16 ...
                 12 11 9 ...
                 8 25 7 6 ...
                 28 3 2 31 ...
                 ];

%% Settings for (5) Hilbert: Output amplitude & phase
hilb_chans = {'PO3';'PO4'}; % Electrodes you want to Hilbert.
hilb_conditions = {'Reset'}; % Events (conditions) of interest.
hilb_freqs = [4:6]; % min/max frequencies of interest e.g., [1 12] is 1 to 12 Hz.
filt_buff = 1; % signal filtered between (current_frequency +/- filt_buff).

%% Settings for (6) Phase-locking value
plv_chans = {'PO3';'PO4'}; % Electrodes you want to PLV.
plv_conditions = {'Reset'}; % Events (conditions) of interest.
plv_freqs = [4:6]; % min/max frequencies of interest e.g., [1 12] is 1 to 12 Hz.

%% Settings for (7) Do Grand
% Plot the ERP's of the current subject only, or all subjects to date!
% Subs to plot; e.g., {'07','08','09', ..., 'ave'}; Put 'ave' at the end!
%subs = {'07','08','09','11','13','14','15','16','18','19','20','22','23','24','25','26','29','30','31','32','33','34','35','36','ave'};
subs = {'02','ave'};

% Declare all independent variables (IV)
% Conditions defining IV (i.e., conds from beh_merge_conds).
iv_conds = [char('current_blocktype','current_reset','current_reset_type')];

% Names for each IV class of interest (part that separates lines in plots or whole plots).
iv_names = [char('shape_nr','shape_sr','shape_cr','color_nr','color_sr','color_cr','both_nr','both_sr','both_cr')];

% Numbers defining the according iv_conds.
% E.g., 10 means iv_conds(1)==1, iv_conds(2)==0.
% n's replace non-applicable iv_conds.
iv_nums = [char('10n','111','112','20n','211','212','30n','311','312')];

%% Settings for (8) Plot ERP Grand
% Channels to plot (unilateral or lateralized)
Unilat_channels = {'Fz';}; % plot_lateralized must NOT == 1.
Contra_channels = {'PO4';'PO8';'P8';'O2';}; % plot_lateralized must == 1.
Ipsi_channels = {'PO3';'PO7';'P7';'O1';}; % plot_lateralized must == 1.

% Name of your ERP, for graph naming purposes.
brainwave_name = ['CDA'];
% Select which IV names (iv_names) will be plotted together;
iv_plots = [1:3; 4:6; 7:9];

% Graph colors
plot_colors = ['k'; 'b'; 'r']; % colors for each line to plot!
plot_magnifier = 1; % magnifies y values for visibility!
plot_dots = [':']; % dot type of dotted lines.

% Saving plots
save_figs = 0; % 1 save plots to new folder, else no saving.
figurefiletype = ['.pdf']; % a filetype; e.g., '.pdf', '.jpg', '.png', etc.

%% Settings for (9) Draw Electrode Surface Map Grand
% Channels to plot (unilateral or lateralized)
surf_map_lateralized = 1; % If 1, plot lateralized (contra/ipsi) in ERP Merge.
surf_map_conditions = [1 2 3 4]; % Plot these conditions. e.g.,[1 2 3] 1 cue, 2 track, 3 reset, etc.
surf_map_conds = [7 8]; % Which conditions in iv_names do you want to plot? e.g., 1 shapenr, 2 shapesr.

% Animation
num_surfmap_loops = 2; % How many times do you want to see the animation?
surf_animation_ratebuffer = .001; % Wait time between each frame in the surfmap animation. Default: '.001'.

% Head/Channels spatial dimensions
head_circumference = 55; % in cm
head_radius = 87.54; % in cm

surf_channels = {'AF3';'F3';'Fz';...
            'FC5';'FC1';'C3';...
            'T7';'P7';'P3';...
            'Pz';'POz';'PO3';'PO7';...
            'O1';'O2';'PO4';'PO8';...
            'P4';'P8';'T8';...%R_Mastoid (Ref) Omitted
            'C4';'Cz';'FC2';'FC6';...
            'F4';'AF4';'AFz';...
            };

% Location of each electrode in [X1 Y1 Z1 ... X2 Y2 Z2 ... X3 Y3 Z3 ... etc.]
% Must be in order of surf_channels!
surf_chan_locs = [-36 76 24; ... %AF3 36 or 26
         -48 59 44; ... %F3
         0 63 61; ... %Fz
         -78 30 27; ... %FC5
         -33 33 74; ... %FC1
         -63 0 61; ... %C3
         -87 0 -3; ...%T7
         -71 -51 -3; ...%P7
         -48 -59 44; ...%P3
         0 -63 61; ...%Pz
         0 -82 31; ...%POz
         -36 -76 24; ...%PO3
         -51 -71 -3; ...%PO7
         -27 -83 -3; ...%O1
         27 -83 -3; ...%O2
         36 -76 24; ...%PO4
         51 -71 -3; ...%PO8
         48 -59 44; ...%P4
         71 -51 -3; ...%P8
         87 0 -3; ...%T8
         63 0 61; ...%C4
         0 0 88; ...%Cz
         33 33 74; ...%FC2
         78 30 27; ...%FC6
         48 59 44; ...%F4
         36 76 24; ...%AF4
         0 82 31; ...%AFz
         ];
     
surf_chan_locs(:,2) = -surf_chan_locs(:,2); % Flip it because before flipping, anterior is down.

%% WARNING! DO NOT EDIT PAST THIS LINE UNLESS YOU WANT TO MAKE DEEP CHANGES!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% (1) Convert Matlab
if sub_data_broken == 0
    sub_folder_name = (cd);
    sub_num = sub_folder_name(length(sub_folder_name)-1:length(sub_folder_name));
    sub_eeg_num = sub_num;
end
%% BEHAVIOURAL: For the number of subject's behavioural files
disp('(1) Converting behavioural files');
% Note, highly experimental section, not automated. EEG and on is good!
% For each behavioural filename present, declare each of the subject's files
sub_eeg_num = char(sub_eeg_num);
sub_beh_num = char([char(sub_beh_prename),char(sub_num)]);

if run_analyses == 1 || run_analyses == 3

    sub_beh_files = 0;
    for beh_file = 1:1:size(sub_beh_num(:,:),1)
        sub_beh_files = sub_beh_files + 1;
        sub_beh_num(beh_file,:) = sub_beh_num(beh_file,:);
        beh_data(beh_file)=load(sub_beh_num(beh_file,~isspace(sub_beh_num(beh_file,:))));
    end
    sub_beh_name = [char(fieldnames(load(sub_beh_num(1,~isspace(sub_beh_num(1,:))))))];

    % If needed, concatenate behavioural files into one
    % % Subject's behavioural file numbers in order; [char('wm_reset_data_15', 'wm_reset_data_15.1', etc.)]
    % sub_beh_num = [char('wm_reset_data_15')]; 
    % sub_beh_name = [char('wm_reset_data')];
    % % Subject's behavioural file trial range in each file.
    % Subject's behavioural file trial range in each file.
    %sub_beh_trials(1,:) = [1:1:720];

    if sub_beh_files > 1
        beh_concat_counter=0;
        for beh_concat = 1:1:size(sub_beh_num,1)-1 % for each behavioural concatenation
            beh_concat_counter = beh_concat_counter+1;
            concat(1) = sub_beh_trials(beh_concat_counter);
            concat(2) = sub_beh_trials(beh_concat_counter+1);
            fnames = fieldnames(beh_data(beh_concat).(sub_beh_name)); % Input the name for subject's behavioural file's struct (i.e., wm_reset_data)
            for f = 1:1:length(fnames)
                current_name = fnames{f};
                current_fieldsize = size(beh_data(beh_concat).(sub_beh_name).(current_name));
                if current_fieldsize(1)==120
                    if length(size(current_fieldsize))==2
                        beh_data(beh_concat).(sub_beh_name).(current_name)(concat_index,:) = temp.(current_name)(:,:);
                    elseif length(size(current_fieldsize))==3
                        beh_data(beh_concat).(sub_beh_name).(current_name)(concat_index,:,:) = temp.(current_name)(:,:,:);
                    elseif length(size(current_fieldsize))==4
                        beh_data(beh_concat).(sub_beh_name).(current_name)(concat_index,:,:,:) = temp.(current_name)(:,:,:,:);
                    end
                elseif current_fieldsize(2)==120
                    if length(size(current_fieldsize))==2
                        beh_data(beh_concat).(sub_beh_name).(current_name)(:,concat_index) = temp.(current_name)(:,:);
                    end
                else
                   if length(size(current_fieldsize))==2
                        beh_data(beh_concat).(sub_beh_name).(current_name)(:,:) = temp.(current_name)(:,:);
                    elseif length(size(current_fieldsize))==3
                        beh_data(beh_concat).(sub_beh_name).(current_name)(:,:,:) = temp.(current_name)(:,:,:);
                    elseif length(size(current_fieldsize))==4
                        beh_data(beh_concat).(sub_beh_name).(current_name)(:,:,:,:) = temp.(current_name)(:,:,:,:);
                   end
                end
            end
        end
        disp('Saving Behaviour');
        save((sub_beh_num(1,:)),(sub_beh_name(1,:)),'-v7.3');
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% BEGIN EEG AND BEHAVIOURAL REFINEMENT
    disp('Loading and converting EEG files');
    % For each filename present, declare each of the subject's header/marker files
    sub_eeg_files = 0;
    for file_num = 1:1:size(sub_eeg_num,1)
        sub_eeg_files = sub_eeg_files+1;
        hdr_num(file_num,:) = [sub_eeg_num(file_num,:),'.vhdr']; % Subject's header file.
        mrk_num(file_num,:) = [sub_eeg_num(file_num,:),'.vmrk']; % Subject's marker file.
    end

    %% EEG: For the number of subject's eeg files
    for eeg_file = 1:1:sub_eeg_files
        % Descriptives; sampling rate, channels, meta (about the data)
        clear fs;
        clear chan;
        clear meta;
        [fs, chan, meta] = bva_readheader(hdr_num(eeg_file,~isspace(hdr_num(eeg_file,:))));

        % Load the series of eeg data; 
        % (31 channels) * (# of data samples)
        clear eeg;
        eeg = bva_loadeeg(hdr_num(eeg_file,~isspace(hdr_num(eeg_file,:))));

        % Load the series of markers; 
        % (marker number, data sample marked) * (# of markers)
        clear temp_markers;
        temp_markers = bva_readmarker(mrk_num(eeg_file,~isspace(mrk_num(eeg_file,:))));
        
        % In case the ???????????
        if size(temp_markers,1)==1 %Unwrap!
            for segment=1:1:size(temp_markers,2)
                if segment==1
                    markers=temp_markers{:,segment};
                else
                    markers=[markers temp_markers{:,segment}];
                end
            end
        else
            markers=temp_markers;
        end

        % Define the harmonized EEG file and its values
        if sub_eeg_files == 1
            EEG.markers=markers;
            EEG.chan=channels;
            EEG.fs=fs;
            %EEG.cell_chan=chan;
            EEG.meta=meta;

            % Filter the eeg data
            l_mastoid_data=eeg(10,:);
            %ave_ref_data=zeros(size(eeg));
            ave_ref_data=eeg-1/2*repmat(l_mastoid_data,[length(chan),1]);
            %filtered_data=zeros(size(eeg));
            %filtered_data=eegfilt(double(ave_ref_data),double(EEG.fs),0.05,0);
            filtered_data=eegfilt(double(ave_ref_data),double(EEG.fs),0,30);
            %EEG.data=double(ave_ref_data);
            EEG.data=filtered_data;

            %disp('Saving EEG');
            %save('EEG.mat','EEG','-v7.3');
        else
            EEGm(eeg_file).markers=markers;
            EEGm(eeg_file).chan=chan;
            EEGm(eeg_file).fs=fs;
            %EEGm(eeg_file).cell_chan=chan;
            EEGm(eeg_file).meta=meta;

            clear l_mastoid_data;
            clear ave_ref_data;
            clear filtered_data;
            
            % Perform eegfilt.m on the eeg data
            l_mastoid_data = eeg(10,:);
            %ave_ref_data = zeros(size(eeg));
            ave_ref_data = eeg-1/2*repmat(l_mastoid_data,length(chan),1);
            %filtered_data = zeros(size(eeg));
            %filtered_data = eegfilt(double(ave_ref_data),double(EEGm(eeg_file).fs),0.05,0);
            filtered_data = eegfilt(double(ave_ref_data),double(EEGm(eeg_file).fs),0,30);
            %EEG1.data = double(ave_ref_data);
            EEGm(eeg_file).data = filtered_data(:,:);

            % Save the EEG partial data
            EEGm_mat_name = ['EEGm',char(string(eeg_file)),'.mat'];
            EEGm_name = ['EEGm',char(string(eeg_file))];

            %save(EEGm_mat_name,EEGm_name,'-v7.3');
        end
    end
    % If multiple subject eeg files, merge them all.
    if sub_eeg_files > 1
        % Fix for the size of all markers.
        EEG_part_sizes = 0;
        for EEGpart = 1:1:sub_eeg_files-1
            EEG_part_sizes = EEG_part_sizes + size(EEGm(EEGpart).data,2);
        end
        EEGm(sub_eeg_files).markers(2,:) = EEGm(sub_eeg_files).markers(2,:) + EEG_part_sizes;
        
        % Merge the EEG files.
        temp_data = [];
        temp_markers = [];
        for EEGpart = 1:1:sub_eeg_files
            temp_markers = [temp_markers EEGm(EEGpart).markers];
            EEG.chan = EEGm(1).chan';
            EEG.fs = EEGm(1).fs;
            %EEG.cell_chan = EEGm(1).cell_chan; 
            EEG.meta = EEGm(1).meta;
            temp_data = [temp_data EEGm(EEGpart).data];
        end
        
        EEG.markers = temp_markers;
        EEG.data = temp_data;
        
        %disp('Saving EEG');
        %save('EEG.mat','EEG','-v7.3');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Summarize behavioural results!
    disp('---------------------------------------------------------');
    disp('------------ Displaying behavioural results! ------------');
    disp('---------------------------------------------------------');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if length(beh_accrt_nums) == 2
        beh_accrt_binary = 1; % If behavioural accuracy is binary (correct/incorr)
    else
        beh_accrt_binary = 0; % If behavioural accuracy is some binomial
    end
    if beh_accrt_binary == 1 % If accrt is binary (correct/incorrect)
        disp('Behavioural accuracy');
        disp([num2str(length(find(beh_data.(sub_beh_name).(char(beh_accrt_conds(1)))==beh_accrt_nums(1)))/length(beh_data.(sub_beh_name).(char(beh_accrt_conds(1))))*100),' %']);
        disp('Behavioural mean RT');
        disp([num2str(mean(beh_data.(sub_beh_name).(char(beh_accrt_conds(2))))*1000),' ms']);
    else % If accuracy is some binomial distribution
        disp('Behavioural accuracy');
        disp([num2str(length(find(beh_data.(sub_beh_name).(char(beh_accrt_conds(1)))==beh_accrt_nums(1)))/length(beh_data.(sub_beh_name).(char(beh_accrt_conds(1))))*100),' %']);
        disp('Behavioural mean RT');
        disp([num2str(mean(beh_data.(sub_beh_name).(char(beh_accrt_conds(2))))*1000),' ms']);
    end
    disp(' ');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% (2) Artifact Detection
    disp('(2) Artifact detection');
    %% Blink detection
    disp('Detecting blinks');
    % Find VEOG from channels index
    blink_chan_counter = 1;
    for current_chan = 1:1:length(EEG.chan)
        curr_chan_name = char(EEG.chan(current_chan,:));
        if isequal(curr_chan_name,'VEOG')==1
            blink_chan_index(blink_chan_counter) = current_chan;
            blink_chan_counter = blink_chan_counter+1;
        end
    end
 
    % Detect blinks (amplitude max-min over time 'window_size') for the 'blink' channels (VEOG)
    blink = zeros(1,size(EEG.data,2));
    for current_chan = 1:1:length(blink_chan_index)
        blink_data = EEG.data(blink_chan_index(current_chan),:);
        t_init = 1;
        while 1
            t_end = t_init+window_size;
            current_data = blink_data(t_init:t_end);
            max_amp = max(current_data);
            min_amp = min(current_data);
            min2max = abs(max_amp-min_amp);
            % flag the blink
            if min2max > blink_threshold
                blink(t_init:t_end) = 1;
            end
            t_init = t_init + step_size;
            % break loop if time window reaches end of EEG data
            if t_init + window_size > length(blink_data)
                break
            end
        end
    end

    %% Eye movement detection
    disp('Detecting eye movements');
    % Find LHEOG and RHEOG from channels index
    eMove = zeros(1,size(EEG.data,2));
    for chan = 1:1:length(EEG.chan)
        curr_chan_name = char(EEG.chan(chan,:));
        if isequal(curr_chan_name,'L_HEOG')==1
            lHEOG_channel_index = chan;
        elseif isequal(curr_chan_name,'R_HEOG')==1
            rHEOG_channel_index = chan;
        end
    end

    % Detect eye movements (amplitude change over time 'window_size') for the 'eMove' channels (LHEOG, RHEOG)
    eMove_data = EEG.data(lHEOG_channel_index,:)-EEG.data(rHEOG_channel_index,:);%lHEOG-rHEOG
    t_init = 1;
    while 1
        t_end = t_init+window_size;
        current_data = eMove_data(t_init:t_end);
        window_center = round(window_size/2);
        pre_amp = mean(current_data(1:window_center-1));
        post_amp = mean(current_data(window_center:length(current_data)));
        % flag the eye movement
        if abs(pre_amp-post_amp)>eMove_threshold
            eMove(t_init:t_end) = 1;
        end
        t_init = t_init + step_size;
        % break loop if time window reaches end of EEG data
        if t_init + window_size > length(eMove_data)
            break
        end
    end

    %% Blocking detection
    disp('Detecting blocking');
    % Detect blocking (amplitude max-min over time 'window_size') for all
    % channels
    blocking = zeros(length(channels),size(EEG.data,2));
    for channel = 1:1:length(channels)%all of them!
        blocking_data = EEG.data(channel,:);
        blocking_count = 0;
        t_init = 1;
        while 1
            t_end = t_init+window_size;
            current_data = blocking_data(t_init:t_end);
            max_amp = max(current_data);
            min_amp = min(current_data);
            % flag the blocking
            if max_amp-min_amp <= blocking_threshold
                blocking(channel,t_init:t_end) = 1;
            end
            t_init = t_init + step_size;
            % break loop if time window reaches end of EEG data
            if t_init + window_size > length(blocking_data)
                break
            end
        end
    end

    % Now save the artifact flags back on EEG
    EEG.arf.blink = blink;
    EEG.arf.eMove = eMove;
    EEG.arf.blocking = blocking;

    if save_raw_data_files == 1
        disp('Saving EEG');
        save('EEG.mat','EEG','-v7.3');
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% (3) Modular; Do_exp_EEG
    disp('(3) Modular; Do_exp_EEG');
    %% Create time windows for each condition
    num_conds = length(conditions);
    disp('Organizing conditions');
    %% Mark each event code of interest
    EEG.trialcodes = zeros(1,length(EEG.data));
    EEG.trialnums = zeros(1,length(EEG.data));
    for ec = 1:1:size(EEG.markers,2) % For each event code.
        if ec+2+length(cond_ecs) > length(EEG.markers)
            break;
        end
        if EEG.markers(1,ec) >= block_ecs(1) && EEG.markers(1,ec) <= block_ecs(2) && EEG.markers(1,ec+2+length(cond_ecs)) == final_ecs(1) %coherent block
            if EEG.markers(1,ec+1) >= trial_ecs(1) && EEG.markers(1,ec+1) <= trial_ecs(2) %trial
                trial_counter = (EEG.markers(1,ec)-block_ecs(1))*numtrialsperblock+EEG.markers(1,ec+1);

                % For each condition, acquire their onset data sample, and mark
                % it down as trialcode (code of condition in trial) and
                % trialnums (whether trial has the condition (1) or not(2)).
                % Replace this if my code is bad, with previous code (below)
                cond_pass = 1;
                for cond = 1:1:num_conds
                    if cond_pass == 1
                        if EEG.markers(1,ec+1+cond) >= cond_ecs(cond,1) && EEG.markers(1,ec+1+cond) <= cond_ecs(cond,2) %condition onset
                            EEG.trialcodes(EEG.markers(2,ec+1+cond)) = cond;
                            EEG.trialnums(EEG.markers(2,ec+1+cond)) = trial_counter;
                        else
                            cond_pass = 0;
                        end
                    end
                end
            end
        end
    end

    %% Separate valid from invalid (artifact) conditions
    disp('Splitting valid from invalid (artifact) conditions');
    trial_counters = zeros(1,length(conditions));
    arf_counter = zeros(4,length(conditions));
    for ec = 1:1:length(EEG.data)
        if EEG.trialcodes(ec) > 0 %found a trialcode concurrent at a data sample!
                trial_counters(EEG.trialcodes(ec)) = trial_counters(EEG.trialcodes(ec))+1;
                fieldname = (char(conditions(EEG.trialcodes(ec),:)));
                trialname = [fieldname,'_trials'];
                artifact = 0;

                % Determine the timewindow depending on which condition it is
                pre_timepoint = cond_timewindows(EEG.trialcodes(ec),1);
                post_timepoint = cond_timewindows(EEG.trialcodes(ec),2);
                pre_timepoint_buff = pre_timepoint+buffer_size;%add buffer to timewindow!
                post_timepoint_buff = post_timepoint+buffer_size;%add buffer to timewindow!

                % Check the time range to see if any artifacts are currently present
                % then note them in artifact counter arf_counter(arftype,condition)
                if sum(EEG.arf.blink(ec-pre_timepoint:ec+post_timepoint))>0
                    artifact = 1;
                    arf_counter(1,EEG.trialcodes(ec)) = arf_counter(1,EEG.trialcodes(ec))+1;
                end
                if sum(EEG.arf.eMove(ec-pre_timepoint:ec+post_timepoint))>0
                    artifact = 1;
                    arf_counter(2,EEG.trialcodes(ec)) = arf_counter(2,EEG.trialcodes(ec))+1;
                end
                if sum(sum(EEG.arf.blocking(:,ec-pre_timepoint:ec+post_timepoint)))>0
                    artifact = 1;
                    arf_counter(3,EEG.trialcodes(ec)) = arf_counter(3,EEG.trialcodes(ec))+1; 
                end

                % Differentiate artifact/clean conditions and at which trial
                if artifact == 0
                    arf_counter(4,EEG.trialcodes(ec)) = arf_counter(4,EEG.trialcodes(ec))+1; % arf 4 is artifact-free.
                    EEG.trials.(fieldname)(trial_counters(EEG.trialcodes(ec)),:,:) = EEG.data(:,ec-pre_timepoint_buff:ec+post_timepoint_buff);
                elseif artifact > 0
                    EEG.trials.(fieldname)(trial_counters(EEG.trialcodes(ec)),:,:) = nan(31,pre_timepoint_buff+post_timepoint_buff+1);
                end
                EEG.trials.(trialname)(trial_counters(EEG.trialcodes(ec))) = EEG.trialnums(ec);
        end
    end

    %% Now create the ERPs and MEANs for each CONDITION and CHANNEL
    disp('Creating ERPs and means for each condition x channel');
    % Create them in (# trials, # datasamples) format. 
    exp_EEG.chan = EEG.chan;
    exp_EEG.buffer_size = buffer_size;
    for condition = 1:1:length(conditions) %name the conditions.
        conditionname = char(conditions(condition,:));
        trialname = [conditionname,'_trials'];
        current_trial = EEG.trials.(trialname);

        % Create CONDITION_CHANNEL indeces in exp_EEG data
        for channel = 1:1:length(channels) 
            channelname = char(channels(channel,:));
            trialchanname = [conditionname,'_',channelname];

            % IMPORTANT! Control for overlap b/c temp_data is 'transparent' like .png
            % since it is to write in place of NANs (see below).
            clear current_data; 
            clear temp_data; 

            current_data = squeeze(EEG.trials.(conditionname)(:,channel,:));
            temp_data = nan(size(current_data,1),size(current_data,2));

            % Create all VALID data sample points in place of NANs. 
            % NANs are therefore INVALID data samples (trial-by-trial).
            temp_data(current_trial,:) = current_data;

            % Formally declare the exp_EEG data!
            exp_EEG.(trialname) = [1:720];
            exp_EEG.(trialchanname) = temp_data;
        end
    end

    %% Second level filter for amplitude RANGE!
    disp('Detecting amplitude range violations');
    range_counter = zeros(1,length(conditions));
    for condition = 1:1:length(conditions)

        % IMPORTANT! Control for overlap b/c these are 'transparent' like .png
        % since it is to write in place of NANs (see below).
        clear range_trial;
        clear unique_range_trial;
        clear max_value;
        clear max_index;
        clear min_value;
        clear min_index;
        clear temp_max_index;
        clear temp_min_index;
        clear temp_range_index;
        conditionname = char(conditions(condition,:));
        range_trial = [];

        for channel = 1:1:length(channels)
            channelname = char(channels(channel,:));
            trialchanname = [conditionname,'_',channelname];

            % IMPORTANT! Control for overlap b/c this is 'transparent' like .png
            % since it is to write in place of NANs (see below).
            clear current_data
            current_data = exp_EEG.(trialchanname);

            % Find MIN and MAX values (w/ reference to mean amplitude for all trials) 
            % within the non-buffered CONDITION_CHANNEL TIMEWINDOWS for each trial, 
            % and at what sample point in that TRIAL.
            [min_value,min_index] = min(current_data(:,buffer_size+1:size(current_data,2)-buffer_size)-repmat(mean(current_data(:,buffer_size+1:size(current_data,2)-buffer_size),2),1,size(current_data,2)-buffer_size*2),[],2);
            [max_value,max_index] = max(current_data(:,buffer_size+1:size(current_data,2)-buffer_size)-repmat(mean(current_data(:,buffer_size+1:size(current_data,2)-buffer_size),2),1,size(current_data,2)-buffer_size*2),[],2);

            % Detect above/below threshold values for each CONDITION_CHANNEL per TRIAL.
            temp_min_index = find(abs(min_value)>range_threshold)';
            temp_max_index = find(abs(max_value)>range_threshold)';
            temp_range_index = unique([temp_max_index temp_min_index]);

            % Collect all trials where range threshold is compromised
            range_trial = [range_trial temp_range_index];
        end

        % If range_trial is NOT EMPTY, store where range is compromised, per
        % CONDITION_CHANNEL.
        if isempty(range_trial) == 0

            % Replace all range compromised trials in exp_EEG as NANs
            unique_range_trial = unique(range_trial);
            range_counter(condition) = length(unique(range_trial));
            for channel = 1:1:length(channels)
                channelname = char(channels(channel,:));
                trialchanname = [conditionname,'_',channelname];
                nan_data = nan(length(unique_range_trial),size(current_data,2));
                exp_EEG.(trialchanname)(unique_range_trial,:) = nan_data;
            end
        end
    end

    %% Second level filter for noise at beginning/end of conditions.
    disp('Detecting tail-end noise');
    noise_counter = zeros(1,length(conditions));
    for condition = 1:1:length(conditions)
        condition_name = (char(conditions(condition,:)));
        noise_index = [];
        for channel = 1:1:length(channels)
            channel_name = char(channels(channel,:));
            dataname = [condition_name,'_',channel_name];

            % IMPORTANT! Control for overlap b/c these are 'transparent' like .png
            % since it is to write in place of NANs (see below).
            clear current_data
            clear temp_index

            current_data = exp_EEG.(dataname);
            temp_index = find(abs(current_data(:,buffer_size+1))>=noise_criteria | abs(current_data(:,size(current_data,2)-buffer_size))>=noise_criteria);
            noise_index = [noise_index; temp_index];
        end

        % If range_trial is NOT EMPTY, store where range is compromised, per
        % CONDITION_CHANNEL.
        if isempty(range_trial) == 0

            % Replace all noise compromised trials in exp_EEG as NANs
            noise_index = unique(noise_index);
            noise_counter(condition) = length(noise_index);
            for channel = 1:1:length(channels)
                channel_name = channels{channel};
                fieldname = [condition_name,'_',channel_name];
                nan_data = nan(length(noise_index),size(exp_EEG.(fieldname),2));
                exp_EEG.(fieldname)(noise_index,:) = nan_data;
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Summarize and display all artifacts!
    disp('---------------------------------------------------------');
    disp('----------- Displaying clean/artifact results -----------');
    disp('---------------------------------------------------------');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cond_counter = 0;
    for condition = 1:1:length(conditions)
        cond_counter = cond_counter+1;

        % Name and declare all artifacts (and artifact-free)
        condition_name = char(conditions(condition,:));
        clean_art_name = [condition_name,'_proportion_clean_trials'];
        art_free_name = [condition_name,'_art_free'];
        blink_name = [condition_name,'_blink'];
        eMove_name = [condition_name,'_eMove'];
        blocking_name = [condition_name,'_blocking'];
        range_name = [condition_name,'_range'];
        noisy_name = [condition_name,'_noisy'];
        exp_art.(art_free_name)= arf_counter(4,condition)-range_counter(condition)-noise_counter(condition);
        exp_art.(blink_name)= arf_counter(1,condition);
        exp_art.(eMove_name)= arf_counter(2,condition);
        exp_art.(blocking_name)= arf_counter(3,condition);
        exp_art.(range_name)= range_counter(condition);
        exp_art.(noisy_name)= noise_counter(condition);

        disp_art.(clean_art_name) = exp_art.(art_free_name)/(numtrialsperblock*numblocks);

        % Condition by timewindow info; add to EEG dataframe
        condtimewindow_name = [condition_name,'_timewindow'];
        EEG.(condtimewindow_name) = cond_timewindows(cond_counter,:);
    end

    disp(disp_art);

    % Save Reset EEG and Reset Artifacts!
    if save_raw_data_files == 1
    disp('Saving ERP');
        save('exp_EEG.mat','exp_EEG','-v7.3');
        save('exp_art.mat','exp_art','-v7.3');
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% (4) Merge; ERP and Behavioural files
    disp('(4) Merging ERP and behaviour');
    %% Declare behavioural
    % Declare the named behavioural conditions to merge with ERP
    exp_EEG_lm.channels = channels;
    for beh_merge_cond = 1:1:size(beh_merge_conds,1)
        exp_EEG_lm.((beh_merge_conds(beh_merge_cond,~isspace(beh_merge_conds(beh_merge_cond,:))))) = beh_data.(sub_beh_name).(beh_merge_conds(beh_merge_cond,~isspace(beh_merge_conds(beh_merge_cond,:))));
    end

    % Find ERP data according to left/right stim presentations (from beh)
    l_index = find(exp_EEG_lm.current_lr == 1);
    r_index = find(exp_EEG_lm.current_lr == 2);

    % Lateralize the ERP data
    for condition = 1:1:length(conditions) % For each condition of interest
        condition_name = char(conditions(condition,:));
        temp_name = [condition_name,'_',char(channels(1))];
        temp_lr_data = nan(size(exp_EEG.(temp_name)));
        for channel = 1:1:length(channels) % For each channel
           channel_name = char(channels(channel,:));
           channel_name_r = char(channels(channel_index_r(channel),:));
           l_data_name = [condition_name,'_',channel_name];
           r_data_name = [condition_name,'_',channel_name_r];

           % Organize temp_lr_data as left/right stim pres
           temp_lr_data(l_index,:) = exp_EEG.(l_data_name)(l_index,:);
           temp_lr_data(r_index,:) = exp_EEG.(r_data_name)(r_index,:);

           % Declare exp_EEG_lm data as orgznied l/r data
           exp_EEG_lm.(l_data_name) = temp_lr_data;
        end
    end

    % Save the merged/lateralized ERP and behaviour
    disp('Saving merged/lateralized ERP and behaviour');
    save('exp_EEG_lm.mat','exp_EEG_lm','-v7.3');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% (5) Hilbert: Output amplitude & phase
    disp('(5) Hilbert: Output amplitude & phase');
    
if run_hilbert == 1
    load exp_EEG_lm;
    % Band-pass transform trial-by-trial
    for condition = 1:1:length(hilb_conditions) % For each condition of interest
        condition_name = char(hilb_conditions(condition,:));
        for channel = 1:1:length(hilb_chans) % For each channel
            channel_name = char(hilb_chans(channel,:));
            data_name = [condition_name,'_',channel_name];
            amp_name = [condition_name,'_',channel_name,'_amp'];
            phase_name = [condition_name,'_',channel_name,'_phase'];
            
            % Clear just in case data overwrite is incomplete (due to NaNs)
            clear curr_data
            clear curr_0mean_data
            clear inst_amp_data
            clear inst_phase_data
            
            % Loop items
            curr_data = exp_EEG_lm.(data_name);
            curr_0mean_data = exp_EEG_lm.(data_name)-nanmean(exp_EEG_lm.(data_name));
            inst_amp_data = zeros(length(1:hilb_freqs(length(hilb_freqs))),size(curr_data,1),size(curr_data,2));
            inst_phase_data = zeros(length(1:hilb_freqs(length(hilb_freqs))),size(curr_data,1),size(curr_data,2));
            % For each frequency of interest ** NOTE: ALIGN EULER_R TO DPS!
            for freq = hilb_freqs(1):hilb_freqs(length(hilb_freqs))
                disp(['Hilbert ',data_name,' ',num2str(freq),' Hz'])
                clear filt_data
                % Band-pass filter the data using Finite Impulse Response
                % filter.
                filt_data = eegfilt(curr_0mean_data(:,:), fs, freq-filt_buff, freq+filt_buff);
                % For each trial
                for trial = 1:size(curr_data,1)
                    % Hilbert transform the data, derive the REAL and
                    % IMAGINARY components of it at each timepoint.
                    clear h_data
                    h_data = hilbert(curr_0mean_data(trial,:));
                    real_h = real(h_data); % X-axis on quadrant
                    imag_h = imag(h_data); % Y-axis on quadrant
                    
                    % Instantaneous Amplitude (Power) and Phase time
                    inst_amp = sqrt(power(real_h,2) + power(imag_h,2)); % Radius of h-value on quadrant
                    inst_phase = atan2(real_h, imag_h); % Angle of h-value on quadrant
                    
                    % Save Inst Amp/Phase data
                    inst_amp_data(freq,trial,:) = inst_amp;
                    inst_phase_data(freq,trial,:) = inst_phase;
                end
            end
            exp_hilb_lm.(amp_name)(:,:,:) = inst_amp_data(:,:,:);
            exp_hilb_lm.(phase_name)(:,:,:) = inst_phase_data(:,:,:);
        end
    end
    disp('Saving hilbert amplitude/phase data');
    save('exp_hilb_lm.mat','exp_hilb_lm','-v7.3');
end

% plot(squeeze(exp_hilb_lm.Reset_PO3_amp(4,200,:)))
% hold
% plot(squeeze(exp_hilb_lm.Reset_PO4_amp(4,200,:)))
% plot(squeeze(exp_hilb_lm.Reset_PO3_phase(4,200,:)))
% plot(squeeze(exp_hilb_lm.Reset_PO4_phase(4,200,:)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% (6) Phase-locking value
    disp('(6) Phase-locking value');

if run_plv == 1
    if save_raw_data_files == 1
        load exp_hilb_lm;
    end
    % Do PLV
    for condition = 1:1:length(plv_conditions) % For each condition of interest
        condition_name = char(plv_conditions(condition,:));
        for send_channel = 1:1:length(plv_chans) % For each sending channel
            send_channel_name = char(plv_chans(send_channel,:));
            send_data_name = [condition_name,'_',send_channel_name];
            send_amp_name = [condition_name,'_',send_channel_name,'_amp'];
            send_phase_name = [condition_name,'_',send_channel_name,'_phase'];
            send_send_name = [condition_name,'_',send_channel_name,'_s'];
            for receive_channel = send_channel+1:1:length(plv_chans) % For each receiving channel
                if strcmp(plv_chans(receive_channel),send_channel_name) == 0
                    receive_channel_name = char(plv_chans(receive_channel,:));
                    receive_data_name = [condition_name,'_',receive_channel_name];
                    receive_amp_name = [condition_name,'_',receive_channel_name,'_amp'];
                    receive_phase_name = [condition_name,'_',receive_channel_name,'_phase'];
                    receive_receive_name = [condition_name,'_',receive_channel_name,'_r'];
                    disp('Phase-locking value')
                    disp(['Sending ',send_channel_name,' Receiving ',receive_channel_name])
                    exp_plv_lm.(send_send_name).(receive_receive_name) = zeros(length(1:plv_freqs(length(plv_freqs))),size(exp_hilb_lm.(receive_amp_name),2),size(exp_hilb_lm.(receive_amp_name),3));
                    for freq = plv_freqs(1):plv_freqs(length(hilb_freqs))
                        for curr_trial = 1:size(exp_hilb_lm.(receive_amp_name),2)
                            for surr_trial = curr_trial+1:size(exp_hilb_lm.(receive_amp_name),2)
                                exp_plv_lm.(send_send_name).(receive_receive_name)(freq,curr_trial,:) = (exp_hilb_lm.(send_phase_name)(freq,curr_trial,:)+exp_hilb_lm.(receive_phase_name)(freq,surr_trial,:))/2;
                                
                            end
                        end
                    end
                end
            end
        end
    end
    disp('Saving Phase-locking value data');
    save('exp_plv_lm.mat','exp_plv_lm','-v7.3');
end

%     %% Step (1) Band-pass filter
%     %% Create the filter
%     % filter equation gaussian=(1/filt_sd*sqrt(2*pi))^-(1/2*((filt_x-filt_mean)/filt_sd)^2)
%     % filter equation gabor wavelet=
%     % filter equation morlet wavelet=
%     % filter parameters
%     filt_sd = .5; % standard deviation of the filter.
%     filt_limits = 2; % +/- this number is the limits of the filter.
%     % freq loop items
%     freq_counter = 0;
%     filt_mean = mean(hilb_freqs);
%     filt_x = zeros(length(hilb_freqs(1):hilb_freqs(length(hilb_freqs))),length(filt_mean-filt_limits:hilb_freq_res:filt_mean+filt_limits));
%     filt_y = zeros(length(hilb_freqs(1):hilb_freqs(length(hilb_freqs))),length(filt_mean-filt_limits:hilb_freq_res:filt_mean+filt_limits));
%     % For each frequency of interest
%     for freq = hilb_freqs(1):hilb_freqs(length(hilb_freqs))
%         freq_counter = freq_counter+1;
%         filt_x_counter = 0;
%         filt_mean = freq;
%         filt_x(freq_counter,:) = filt_mean-filt_limits:hilb_freq_res:filt_mean+filt_limits;
%         % for each x-axis value, set y as per the filter equation
%         for filt_x_loop = filt_mean-filt_limits:hilb_freq_res:filt_mean+filt_limits
%             filt_x_counter = filt_x_counter+1;
%             filt_y(freq_counter,filt_x_counter) = (1/filt_sd*sqrt(2*pi))^-(1/2*((filt_x_loop-filt_mean)/filt_sd)^2); % filter equation here!
%         end
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if run_analyses == 2 || run_analyses == 3

    %% (7) Do Grand
    disp('(7) Creating Grand');
    % For each subject, grab their EEG files and merge with grand EEG files.
    numsubs = length(subs);
    Grandexp_EEG = struct();
    Grandexp_EEG.subs = subs;
    for s = 1:1:numsubs % For each sub, including 'average'.
        sub_name = char(subs(:,s));
        if s < numsubs
            cd ../
            cd (sub_name)

            clear exp_EEG_lm;
            load exp_EEG_lm;

            % Create array where for each iv_name, each trial is 1 or 0 for
            % meeting IV conditions (e.g., blocktype==1, current_reset==1, etc.)
            clear iv_cond_index
            iv_cond_index = zeros(size(iv_names,1),size(iv_conds,1),1,numtrialsperblock*numblocks);
            num_conds = zeros(1,size(iv_names,1));
            for iv_name = 1:1:size(iv_names,1)
                for iv_cond = 1:1:size(iv_conds,1) % iv_name defining IV's
                    if iv_nums(iv_name,iv_cond) == 'n'
                        iv_cond_index(iv_name,iv_cond,:,:) = zeros(1,size(exp_EEG_lm.(iv_conds(iv_cond,~isspace(iv_conds(iv_cond,:)))),2));
                    else
                        num_conds(iv_name) = num_conds(iv_name)+1;
                        iv_cond_index(iv_name,iv_cond,:,:) = exp_EEG_lm.(iv_conds(iv_cond,~isspace(iv_conds(iv_cond,:))))==str2num(iv_nums(iv_name,iv_cond));
                    end
                end
            end

            % For each iv_name, check the sum of the above indeces at each
            % trial. For trials where all conditions per iv_name are met, save
            % them in the iv_total_index.
            clear iv_total_index
            for iv_name = 1:1:size(iv_names,1)
                curr_iv_name = char(iv_names(iv_name,~isspace(iv_names(iv_name,:))));
                iv_total_index.(curr_iv_name) = find(sum(iv_cond_index(iv_name,:,:,:))==num_conds(iv_name))';
            end

            % Create the Grand EEG (Add each sub into the grand moshpit)!
            % This is done by grabbing specific trials based on the IVs (iv_name).
            for condition = 1:1:length(conditions)
                condition_name = char(conditions(condition,:));
                for channel = 1:1:length(channels)
                    channel_name = char(channels(channel,:));
                    dataname = [condition_name,'_',channel_name];
                    basename = ['Cue_',channel_name];
                    ampname = [condition_name,'_',channel_name,'_amp'];
                    for iv_name = 1:1:size(iv_names,1)
                        curr_iv_name = char(iv_names(iv_name,~isspace(iv_names(iv_name,:))));
                        Grandexp_EEG.(dataname)(s,iv_name,:) = squeeze(nanmean(exp_EEG_lm.(dataname)(iv_total_index.(curr_iv_name),buffer_size+1:size(exp_EEG_lm.(dataname),2)-buffer_size),1))-squeeze(mean(nanmean(exp_EEG_lm.(basename)(iv_total_index.(curr_iv_name),buffer_size+1:buffer_size+50),1),2));
                    end
                end
            end

            % Now for phase
            disp(['Adding ',sub_name,' to Grand']);
            cd ../
            cd(sub_name);

        % Store average data of all subs.
        elseif s == numsubs
            for condition = 1:1:length(conditions)
                condition_name = char(conditions(condition,:));
                for channel = 1:1:length(channels)
                    channel_name = char(channels(channel,:));
                    dataname = [condition_name,'_',channel_name];
                    ampname = [condition_name,'_',channel_name,'_amp'];
                    Grandexp_EEG.(dataname)(s,:,:,:) = squeeze(mean(Grandexp_EEG.(dataname)(1:s-1,:,:),1));
                    %Grandexp_EEG.(ampname)(s,:,:,:) = squeeze(mean(Grandexp_EEG.(ampname)(1:s-1,:,:),1));
                end
            end
        end
    end

    % Save the grand file.
    if save_raw_data_files == 1
        disp('Grand creation completed');
        disp(cd);
        save('Grandexp_EEG.mat','Grandexp_EEG','-v7.3');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Establish each graph's X-values (time)
tpers = 1000/fs; % time per sample (if fs = 250hz, then spers = 4).
plot_timewindows = cond_timewindows*tpers; % convert samples timewindows to actual time.
numsubs = length(subs); % # of subjects in grand
amplock_cond_name = char(conditions(amplock_cond));

%% (8) Plot ERP Grand
if run_plot == 1   
    disp('(8) Plotting ERP Grand');
    for curr_cond = 1:1:length(conditions)
        condition_name = char(conditions(curr_cond,:));
        mainplot.(condition_name).('time') = [(-plot_timewindows(curr_cond,1)):tpers:plot_timewindows(curr_cond,2)]';
    end
    if surf_map_lateralized == 1
        plot_channels = length(Contra_channels);
    else
        plot_channels = length(Unilat_channels);
    end

    % Create zeros, establish condition names by contra/ipsi for plotting (plot_x).
    for curr_cond = 1:1:length(conditions)
        condition_name = char(conditions(curr_cond,:));
        channel_name = char(channels(1,:));
        temp_cond_chan_name = [condition_name,'_',channel_name];
        curr_cond_length = size(Grandexp_EEG.(temp_cond_chan_name),3);
        if surf_map_lateralized == 1
            preplot.(condition_name).('contra') = zeros(numsubs,size(iv_names,1),curr_cond_length);
            preplot.(condition_name).('ipsi') = zeros(numsubs,size(iv_names,1),curr_cond_length);
        else
            preplot.(condition_name).('unilat') = zeros(numsubs,size(iv_names,1),curr_cond_length);
        end
    end

    % Apply the Grandexp_EEG data to the zeros, by contra/ipsi.
    for channel = 1:1:plot_channels
        if surf_map_lateralized == 1
            contra_channel_name = char(Contra_channels(channel,:));
            ipsi_channel_name = char(Ipsi_channels(channel,:));
        else
            unilat_channel_name = char(Unilat_channels(channel,:));
        end
        for curr_cond = 1:1:length(conditions)
            condition_name = char(conditions(curr_cond,:));
            if surf_map_lateralized == 1
                contra_cond_name = [condition_name,'_',contra_channel_name];
                ipsi_cond_name = [condition_name,'_',ipsi_channel_name];
                preplot.(condition_name).('contra')(:,:,:) = preplot.(condition_name).('contra')+Grandexp_EEG.(contra_cond_name)(:,:,:);
                preplot.(condition_name).('ipsi')(:,:,:) = preplot.(condition_name).('ipsi')+Grandexp_EEG.(ipsi_cond_name)(:,:,:);
            else
                unilat_cond_name = [condition_name,'_',unilat_channel_name];
                preplot.(condition_name).('unilat')(:,:,:) = preplot.(condition_name).('unilat')+Grandexp_EEG.(unilat_cond_name)(:,:,:);
            end
        end
    end

    % Get the difference waves by doing Contra-Ipsi!!! (This is the CDA <3!)!
    if surf_map_lateralized == 1
        for curr_cond = 1:1:length(conditions)
            condition_name = char(conditions(curr_cond,:));
            preplot.(condition_name).('contra') = preplot.(condition_name).('contra')/(length(Contra_channels));
            preplot.(condition_name).('ipsi') = preplot.(condition_name).('ipsi')/(length(Ipsi_channels));
            preplot.(condition_name).('diff') = preplot.(condition_name).('contra')-preplot.(condition_name).('ipsi');
        end
    end

    % Find the waves WITHIN subjects relative to baseline (amplock); mean waves of all IVs
    % within subjects, for each time (sample) point.
    for sub = 1:1:numsubs % For each subject
        for curr_cond = 1:1:length(conditions) % For each condition
            condition_name = char(conditions(curr_cond,:));
            temp_cond_chan_name = [condition_name,'_',channel_name];
            curr_cond_length = size(Grandexp_EEG.(temp_cond_chan_name),3);
            for iv = 1:1:length(iv_names) % For each IV
                for time = 1:1:curr_cond_length % For each sample (time) point
                    if surf_map_lateralized == 1
                        mainplot.(condition_name).('contra')(sub,iv,time) = preplot.(condition_name).('contra')(sub,iv,time)-squeeze(mean(preplot.(amplock_cond_name).('contra')(sub,iv,cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2))));%-squeeze(mean(preplot.(condition_name).('contra')(sub,:,time),2))+squeeze(mean(mean(preplot.(condition_name).('contra')(1:numsubs-1,:,time),1),2));
                        mainplot.(condition_name).('ipsi')(sub,iv,time) = preplot.(condition_name).('ipsi')(sub,iv,time)-squeeze(mean(preplot.(amplock_cond_name).('ipsi')(sub,iv,cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2))));%-squeeze(mean(preplot.(condition_name).('ipsi')(sub,:,time),2))+squeeze(mean(mean(preplot.(condition_name).('ipsi')(1:numsubs-1,:,time),1),2));
                        mainplot.(condition_name).('diff')(sub,iv,time) = preplot.(condition_name).('diff')(sub,iv,time)-squeeze(mean(preplot.(amplock_cond_name).('diff')(sub,iv,cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2))));%-squeeze(mean(preplot.(condition_name).('diff')(sub,:,time),2))+squeeze(mean(mean(preplot.(condition_name).('diff')(1:numsubs-1,:,time),1),2));
                    else
                        mainplot.(condition_name).('unilat')(sub,iv,time) = preplot.(condition_name).('unilat')(sub,iv,time)-squeeze(mean(preplot.(amplock_cond_name).('unilat')(sub,iv,cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2))));%-squeeze(mean(preplot.(condition_name).('unilat')(sub,:,time),2))+squeeze(mean(mean(preplot.(condition_name).('unilat')(1:numsubs-1,:,time),1),2));
                    end
                end
            end
        end
    end

    % Find the within-subjects error values.
    for curr_cond = 1:1:length(conditions) % For each condition
        condition_name = char(conditions(curr_cond,:));
        if surf_map_lateralized == 1
            mainplot.(condition_name).('contra_se') = std(mainplot.(condition_name).('contra')(1:numsubs-1,:,:),1)/sqrt(numsubs-1);
            mainplot.(condition_name).('ipsi_se') = std(mainplot.(condition_name).('ipsi')(1:numsubs-1,:,:),1)/sqrt(numsubs-1);
            mainplot.(condition_name).('diff_se') = std(mainplot.(condition_name).('diff')(1:numsubs-1,:,:),1)/sqrt(numsubs-1);
        else
            mainplot.(condition_name).('unilat_se') = std(mainplot.(condition_name).('unilat')(1:numsubs-1,:,:),1)/sqrt(numsubs-1);
        end
    end
    
    % Now plot the graphs organized by IV names/conditions (iv_plots).
    disp('Plotting');
    if save_figs == 1
        mkdir figures;
    end
    for curr_cond = 1:1:length(conditions)
        condition_name = char(conditions(curr_cond,:));
        line_counter = 0;
        for curr_plot = 1:1:length(iv_plots) % For each graph to plot
            fig = figure;
            for curr_line = 1:1:length(iv_plots(curr_plot,:)) % For each line to plot.
                line_counter = line_counter+1;
                if surf_map_lateralized == 1
                    realplot(line_counter) = plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('diff')(numsubs,line_counter,:))*plot_magnifier,[plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                    plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('diff')(numsubs,line_counter,:))+squeeze(mainplot.(condition_name).('diff_se')(:,1,:))*plot_magnifier,[plot_dots,plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                    plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('diff')(numsubs,line_counter,:))-squeeze(mainplot.(condition_name).('diff_se')(:,1,:))*plot_magnifier,[plot_dots,plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                else
                    realplot(line_counter) = plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('unilat')(numsubs,line_counter,:))*plot_magnifier,[plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                    plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('unilat')(numsubs,line_counter,:))+squeeze(mainplot.(condition_name).('unilat_se')(:,1,:))*plot_magnifier,[plot_dots,plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                    plot(mainplot.(condition_name).('time'),squeeze(mainplot.(condition_name).('unilat')(numsubs,line_counter,:))-squeeze(mainplot.(condition_name).('unilat_se')(:,1,:))*plot_magnifier,[plot_dots,plot_colors(curr_line)],'LineWidth',2);
                    hold on;
                end
            end
            legend([realplot(line_counter-curr_line+1:line_counter-curr_line+length(iv_plots))],iv_names(line_counter-curr_line+1:line_counter-curr_line+length(iv_plots),:));
            mainplottitle = [upper(brainwave_name),' for ',upper(condition_name),' at IVs - ',upper(iv_names(line_counter-curr_line+1,:))];
            title(mainplottitle);
            xlabel(['Time (t) relative to ',upper(condition_name),' onset']);
            ylabel(['Amplitude (µV) of ',upper(brainwave_name)]);
            if save_figs == 1
                cd figures;
                saveas(fig,[mainplottitle,figurefiletype]);
                cd ..;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% (9) Draw Electrode Surface Map Grand
if run_plot == 2
    
    disp('(9) Draw Electrode Surface Map Grand');
    % Get screen, subject, and port info
    Screen('Preference', 'SkipSyncTests', 1);
    Screensizes = get(0, 'MonitorPositions');
    framerate = round(FrameRate);
    Screensize = Screensizes(1,:);
    screensize = [800 800];
    screencenter = [screensize(1)/2 screensize(2)/2];
    background_color = [200 200 200];
    [wPtr,rect] = Screen(0, 'OpenWindow', [background_color], [0 0 screensize(1) screensize(2)]);
    priorityLevel = MaxPriority(wPtr);
    Priority(priorityLevel);
    Screen(wPtr,'Flip');
    font_size = round(screensize(1)/50);
    Screen('TextSize', wPtr, font_size);

    % Keycodes
    if os == 2 %pc
        SpaceKeyCodes = [32];%for pc
    elseif os == 1 %mac
        SpaceKeyCodes = [44];%for mac
    end
    
    elec_size = screensize(1)*.0225;
    
    % Animation of Electrodes.
    surfmap_counter = 0;
    while surfmap_counter <= num_surfmap_loops
        surfmap_counter = surfmap_counter+1;
        elec_color = zeros(length(surf_chan_locs),3);
        pos_color = [150 0 0];
        neg_color = [0 150 0];
        ctrl_color = pos_color+neg_color;
        for condition = 1:1:length(conditions) % For each condition
            if ismember(condition, surf_map_conditions) == 1 % If the condition is of interest
                condition_name = char(conditions(condition,:));
                temp_name = [condition_name,'_',char(surf_channels(1))];
                Screen('FillOval',wPtr, ctrl_color, [screencenter(1)-150-elec_size screencenter(2)-elec_size screencenter(1)-150+elec_size screencenter(2)+elec_size]);
                Screen('FillRect',wPtr, background_color, [screencenter(1)-screensize(1)/2 screencenter(2)-screensize(2)/2 screencenter(1)+screensize(1)/2 screencenter(2)+screensize(2)/2]);
                Screen('DrawText', wPtr, [condition_name, ' ', iv_names(surf_map_conds(1),:), ' - ', iv_names(surf_map_conds(2),:)], [screencenter(1,1)-40*screensize(1)/350], [screencenter(1,2)-150*screensize(2)/350], [0 0 0]);
                Screen(wPtr, 'Flip');
                WaitSecs(.1);
                for s = 1:1:length(Grandexp_EEG.(temp_name)) % For the # of samples in the EEG data of the condition
                    preelec_baselined_cond1 = zeros(length(Grandexp_EEG.(temp_name)));
                    preelec_baselined_cond2 = zeros(length(Grandexp_EEG.(temp_name)));
                    preelec_baselined_bothconds = zeros(length(Grandexp_EEG.(temp_name)));
                    elec_color = zeros(length(surf_chan_locs),3);
                    for surf_chan = 1:1:length(surf_chan_locs) % For the # of channels we are surf-mapping
                        temp_name = [condition_name,'_',char(surf_channels(surf_chan))];
                        amplock_name = [char(conditions(amplock_cond)),'_',char(surf_channels(surf_chan))];
                        preelec_baselined_cond1(surf_chan) = Grandexp_EEG.(temp_name)(numsubs,surf_map_conds(1),s)-mean(Grandexp_EEG.(amplock_name)(numsubs,surf_map_conds(1),cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2)));
                        preelec_baselined_cond2(surf_chan) = Grandexp_EEG.(temp_name)(numsubs,surf_map_conds(2),s)-mean(Grandexp_EEG.(amplock_name)(numsubs,surf_map_conds(2),cond_timewindows(2,1)-amplock_timewindow(1):cond_timewindows(2,1)+amplock_timewindow(2)));
                        preelec_baselined_bothconds(surf_chan) = (preelec_baselined_cond2(surf_chan)-preelec_baselined_cond1(surf_chan))*25;
                        % Set the color for each electrode for each frame.
                        for generalloop = 1:1:3
                            if pos_color(generalloop) ~= 0
                                elec_color(surf_chan,generalloop) = ctrl_color(generalloop)+preelec_baselined_bothconds(surf_chan);
                            end
                            if neg_color(generalloop) ~= 0
                                elec_color(surf_chan,generalloop) = ctrl_color(generalloop)-preelec_baselined_bothconds(surf_chan);
                            end
                        end
                        % Draw the colored electrodes
                        Screen('FillOval',wPtr, elec_color(surf_chan,:), [screencenter(1)+surf_chan_locs(surf_chan,1)*screensize(1)/350-elec_size screencenter(2)+surf_chan_locs(surf_chan,2)*screensize(2)/350-elec_size screencenter(1)+surf_chan_locs(surf_chan,1)*screensize(1)/350+elec_size screencenter(2)+surf_chan_locs(surf_chan,2)*screensize(2)/350+elec_size]);
                        Screen('DrawText', wPtr, [char(surf_channels(surf_chan))], [screencenter(1)+surf_chan_locs(surf_chan,1)*screensize(1)/350-14], [screencenter(2)+surf_chan_locs(surf_chan,2)*screensize(2)/350-5], [0 0 0]);
                    end
                    % Draw legends/supplementary material
                    if surf_map_lateralized == 1
                        Screen('DrawText', wPtr, ['Left Lateral Target'], [screencenter(1,1)-100*screensize(1)/350], [screencenter(1,2)-120*screensize(2)/350], [0 0 0]);
                        Screen('DrawText', wPtr, ['<< ipsilateral               posterior                contralateral >>'], [screencenter(1,1)-90*screensize(1)/350], [screencenter(1,2)+120*screensize(2)/350], [0 0 0]);
                    else
                        Screen('DrawText', wPtr, ['Unilateral Target'], [screencenter(1,1)-40*screensize(1)/350], [screencenter(1,2)-120*screensize(2)/350], [0 0 0]);
                    end
                    Screen('FillOval',wPtr, pos_color, [screencenter(1)-150*screensize(1)/350-elec_size screencenter(2)-20*screensize(1)/350-elec_size screencenter(1)-150*screensize(1)/350+elec_size screencenter(2)-20*screensize(1)/350+elec_size]);
                    Screen('FillOval',wPtr, ctrl_color, [screencenter(1)-150*screensize(1)/350-elec_size screencenter(2)-elec_size screencenter(1)-150*screensize(1)/350+elec_size screencenter(2)+elec_size]);
                    Screen('FillOval',wPtr, neg_color, [screencenter(1)-150*screensize(1)/350-elec_size screencenter(2)+20*screensize(1)/350-elec_size screencenter(1)-150*screensize(1)/350+elec_size screencenter(2)+20*screensize(1)/350+elec_size]);
                    Screen('DrawText', wPtr, ['+'], [screencenter(1,1)-140*screensize(1)/350], [screencenter(1,2)-20*screensize(2)/350], [0 0 0]);
                    Screen('DrawText', wPtr, ['Ctrl'], [screencenter(1,1)-140*screensize(1)/350], [screencenter(1,2)-0*screensize(2)/350], [0 0 0]);
                    Screen('DrawText', wPtr, ['-'], [screencenter(1,1)-140*screensize(1)/350], [screencenter(1,2)+20*screensize(2)/350], [0 0 0]);
                    
                    Screen('DrawText', wPtr, [condition_name, ' ', iv_names(surf_map_conds(1),:), ' - ', iv_names(surf_map_conds(2),:)], [screencenter(1,1)-90*screensize(1)/350], [screencenter(1,2)-150*screensize(2)/350], [0 0 0]);
                    Screen('DrawText', wPtr, ['t = ', num2str(s*tpers-cond_timewindows(1)*tpers), 'ms / ', num2str(length(Grandexp_EEG.(temp_name))*tpers-cond_timewindows(1)*tpers)], [screencenter(1,1)-90*screensize(1)/350], [screencenter(1,2)-140*screensize(2)/350], [0 0 0]);
                    Screen(wPtr, 'Flip');
                    WaitSecs(surf_animation_ratebuffer);
                    % Press SPACEBAR to skip to next condition.
                    while KbCheck; end
                    [keyIsDown, sec, keyCode] = KbCheck;
                    keyCode = find(keyCode, 1);
                    temp_keyCode = 0;
                    if keyIsDown
                        temp_keyCode = keyCode;
                    end
                    if ismember(temp_keyCode, SpaceKeyCodes)
                        break;
                    end
                end
            end
        end
    end
    Screen('CloseAll');
    sca;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Display critical settings
disp('---------------------------------------------------------');
disp('---------------------------FIN---------------------------');
disp('---------------------------------------------------------');

disp('---------------------------------------------------------');
disp('-------------------Summary of Settings-------------------');
disp('---------------------------------------------------------');
disp(['Subject # ',char(sub_num(1,:))]);
disp(['# of trials: ', num2str(numtrialsperblock*numblocks)]);
disp('Conditions -----');
disp(conditions)
if run_analyses == 2 || run_analyses == 3 || run_plot == 1
    disp(['IV names -----']);
    disp(iv_names);
    disp('Channels -----');
    if surf_map_lateralized == 1
        disp(Contra_channels);
        disp(Ipsi_channels);
    else
        disp(Unilat_channels);
    end
end
disp('-------------------');
disp(['Brainwaves - ',brainwave_name]);
disp('-------------------');
if run_analyses == 2 || run_analyses == 3 || run_plot == 1
    if save_figs == 1
        disp(['Saved figures as ',figurefiletype]);
    else
        disp('Not saving figures');
    end
end

if run_hilbert == 1
    disp('-------------------');
    disp(['FIR Filter buffer: +/- ',num2str(filt_buff),' Hz']);
    disp('-- Hilbert frequencies in Hz --');
    disp(num2str(hilb_freqs));
end
if run_plv == 1
    disp('-------------------');
    disp('-- Phase-locking value frequencies in Hz --');
    disp(num2str(plv_freqs));
end

if run_analyses == 1 || run_analyses == 3
    disp('-------------------');
    if beh_accrt_binary == 1 % If accrt is binary (correct/incorrect)
        disp('Behavioural accuracy');
        disp([num2str(length(find(beh_data.(sub_beh_name).(char(beh_accrt_conds(1)))==beh_accrt_nums(1)))/length(beh_data.(sub_beh_name).(char(beh_accrt_conds(1))))*100),' %']);
        disp('Behavioural mean RT');
        disp([num2str(mean(beh_data.(sub_beh_name).(char(beh_accrt_conds(2))))*1000),' ms']);
    else % If accuracy is some binomial distribution
        disp('Behavioural accuracy');
        disp([num2str(length(find(beh_data.(sub_beh_name).(char(beh_accrt_conds(1)))==beh_accrt_nums(1)))/length(beh_data.(sub_beh_name).(char(beh_accrt_conds(1))))*100),' %']);
        disp('Behavioural mean RT');
        disp([num2str(mean(beh_data.(sub_beh_name).(char(beh_accrt_conds(2))))*1000),' ms']);
    end
    disp(' ');
end

if run_analyses == 1 || run_analyses == 3
    disp('-------------------');
    disp(disp_art);
end

disp('---------------- ^ Summary of settings ^ ----------------');
disp('---------------------------------------------------------');
disp('---------------------------END---------------------------');
disp('---------------------------------------------------------');
disp("Thank you for using Bens Complete ERP Analysis Function");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
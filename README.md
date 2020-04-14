--- Welcome to Ben's Complete ERP Analysis Experience! ---
Hyun Goo (Ben) Park #1001605578
PSY1210 2019-2020
University of Toronto

--- ABOUT THE ANALYSIS ---
This experience currently loads BrainVision EEG data and Matlab behavioural data acquired during an experiment, filters out artifacts from the raw EEG data (like eye-blinks, high frequency noise), merges and organizes the EEG and behavioural data according to user input settings, and visualizes the organized data, all in a user-friendly way.

The aim of this experience is to automate as much of time-signal/frequency analysis as possible, while giving appropriate flexibility to the user - thus a beginner- to expert-friendly interface. This format enables users to analyze any data from any experiment by changing just a few input settings on LINES 73-285 (most of these lines are comments/not needed to be changed)!

--- TO RUN THE EXPERIENCE: ---
1) Please make sure you have Matlab and Psychtoolbox installed.
2) Please open folder '02'.
3) Please open 'PSY1210_Project2_b_cERPa_Apr1120.m' in Matlab.
4) Please set your operating system on LINE 74.

OPTIONAL:
5) You may change the method of visualizing the data (1 = Plot() ERP's, 2 = Animation of
Surface map of electrodes) on LINE 80.
6) Other settings are available to tweak for your testing in LINES 73-285, segregated by chapter.

WHEN READY:
7) RUN THE EXPERIENCE :)!

--- DISCLAIMER ---
Please note, this analysis uses the following functions that are NOT made by me:
- 'bva_loadeeg.m'
- 'bva_readheader.m'
- 'bva_readmarker.m'
- 'eegfilt.m'

Although this analysis code is mainly original work, chapters (1-4; refer to table of contents on LINES 26-39) are heavily inspired by Dr. Keisuke Fukuda's EEG analysis code, and thus the code outline/variable names/values may be similar or identical. Appropriate
Permission was obtained.

--- LIMITATIONS ---
Work in progress: 
- Plotting of (5) Hilbert Transform and (6) Phase-locking value are currently unavailable.
- Can be further automated.

The data used in this experience are not valid - this is pilot data whose experiment was not correctly coded. Thus, the data may not be accurately visualized. However, the experience works as intended, since real data are accurately visualized using the experience.

--- ABOUT THE EXPERIMENT ---
Task-relevant features (e.g., SHAPE during a SHAPE task) are present in visual working memory. We wanted to test whether task-irrelevant features (e.g., COLOUR during a SHAPE task) are also present in visual working memory, by measuring whether visual working memory activity (measured by brainwaves) changes in response to task-irrelevant events, as it does for task-relevant events.
(CUE) Participants are first presented with a static object - a coloured shape.
(TRACK) The static object then begin to dynamically change along their SHAPE and COLOUR dimensions. Participants are instructed in a block-by-block manner to track either the (1) SHAPE, (2) COLOUR, or (3) BOTH COLOUR AND SHAPE of the object as they change, to later report the last (1) shape, (2) colour, or (3) either, presented during the tracking phase.
(RESET) During the tracking phase, the coloured shape could discontinuously change in either its shape or its colour, or not at all. If visual working memory holds task-irrelevant features, then when the current task is 'TRACK SHAPE', a discontinuous change in 'TRACK COLOUR' should still evoke an ERP. Else, we should see no evoked ERP.
(RETENTION INTERVAL) After the tracking phase, objects disappear and a brief retention interval is presented.
(TEST) After the retention interval, participants are presented with a test array consisting of two objects. In the TRACK SHAPE blocks, one of these objects are identical in their shape to the last-presented shape in the tracking phase. The other object is a lure that has a similar, but different shape to the last-presented shape in the tracking phase. In the TRACK COLOUR blocks, two colour objects are presented instead of shapes. In the TRACK BOTH blocks, either two shape, or two colour objects are presented. Participants are required to report which of the two objects were identical in shape or colour to the last-presented object in the tracking phase.

Thus, the EXPERIMENTAL CONDITIONS are as follows:
IV1: Block type of SHAPE, COLOUR or BOTH.
IV2: Discontinuous change PRESENT or ABSENT per trial.
DV1: ERP named the contralateral delay activity, which reflects the amount of information held in the participant's visual working memory.
DV2: Participant's reporting correctness in response to the test array, of the last shape/colour presented during the tracking phase.

--- THANK YOU! :) ---

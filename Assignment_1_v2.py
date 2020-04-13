#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a trial loop Step 1
Use this template script to present one trial with your desired structure
@author: katherineduncan
"""
#%% Required set up 
# this imports everything you might need and opens a full screen window
# when you are developing your script you might want to make a smaller window 
# so that you can still see your console 

import numpy as np
import pandas as pd
import os #, sys
import shutil
import psychopy
from psychopy import visual, core, event, gui, logging
import random

###############################
#### A. Experiment Details ####
###############################
# PROJECT 1 CONCERNS ONLY 1A.
# 1a) Participants track a moving stimulus with their mouse, to the best of their ability.
# The experimental question is whether a reset in stimulus location before tracking results in a different
# attentional signature (i.e., changed mouse-stimulus tracking accuracy, or weaker 8hz to other hz as
# measured by fourier transform).
# Please ensure in B. Quick Settings that debug, debug_fps, trial_analysis, trial_prediction, playback,
# and full_analysis are set to 0 before running participants.

# Note: My goal is to have trial-by-trial fourier and trial-by-trial frame-by-frame accuracy 
# correlated, then provide trial-by-trial feedback of predicted attentional state (good/bad)!

# 2) Cue gives participants mean velocity information.
# BUT! Trial can end at any time, and must report location after RI. Click on where
# stimulus was.
# 2) Analysis; Take frequency. Should both be 8 Hz. But, 8 Hz should be at
# different points of cycle. How to parse this?
# Analyze at what time points the endogenous and exogenous Hz occur.
# Are they significantly different at any time points when compared?
# Is their difference between frequency across time also at 4 Hz?
# Then this would suggest they are cycling.

# 2) Can feedback help tracking performance? Predict trial-by-trial accuracy
# using heatmap/fourier.
#
# TO DO
# Above, and add a acc_rad visualization to heatmap. Line radius = mean acc_rad,
# and circles for SD of acc_rad. 95% confidence interval?

###########################
#### B. Quick Settings ####
###########################
subnum=1 #subject number
# Note: Position is always defined as maximum of .5, and in coordinate plane.
fullscreen=0 #fullscreen win or not?
debug=1 #Debug mode on (1) or off (0).
debug_fps=0 #shows fps data
trial_analysis=0 #Trial-by-trial analysis, used for model-prediction.
trial_analysis_visual=0 # If trial_analysis is 1, and if you want to visualize trial_analysis.
trial_prediction=0 #Within-trial prediction of accuracy (acc_rad).
trial_prediction_visual=0 # see the trial prediction graphs.
frame_prediction=0 #Accuracy prediction frame-by-frame per trial.
frame_prediction_visual=0 #if frame_prediction is 1, and you want to visualize frame_prediction.
playback=0 #playback the trial after trial_analysis.
full_analysis=0 #for later, when we analyze the entire experiment!
trail_visual=0 #calculate and see the trail (acceleraion amount per frame)

# fourier resolution
fourier_min_freq=1 #minimum frequency we should fourier (minimum is == fourier_freq_res).
fourier_max_freq=12 #maximum frequency we should fourier.
fourier_freq_res=.5 #resolution of fourier transform; min to max in this interval. Must be divisible by 1.

# open a white full screen window
screen_x=1920  #5120 iMac
screen_y=1080  #2880 iMac
framerate=60

if debug==1:
    screen_x=800
    screen_y=800
    framerate=60

# background color
background_color='black'

time_per_frame=1/framerate
time_per_frame_precision=time_per_frame/4

if fullscreen==1:
    win = psychopy.visual.Window(size=[screen_x,screen_y],fullscr=True, allowGUI=True, color=background_color, units='height')
else:
    win = psychopy.visual.Window(size=[screen_x,screen_y],fullscr=False, allowGUI=True, color=background_color, units='height')

#%% up to you!
# this is where you build a trial that you might actually use one day!
# just try to make one trial ordering your lines of code according to the 
# sequence of events that happen on one trial
# if you're stuck you can use the responseExercise.py answer as a starting point 

#%% Notes
#%% To Do:
# Save data
# Heading direction data
# Variable speeds, accelerations, etc.
# Difference b/w mouse_pos and stim_pos depending on heading direction of stimulus
# Subject inputs, data saving, moving, etc.
# Instructions?

#%% Experiment Parameters
#block and trial settings
numblocks=1
numtrialsperblock=2
#mouse tracking
if debug==1:
    mouse=event.Mouse(visible=True,win=win)
else:
    mouse=event.Mouse(visible=False,win=win)
#object parameters
c1_rad=.01
#physical limits
spdlmt=.005
acclmt=1
poslmt=.5
startpos_buf=.1
acc_multiplier=.001
opp_acc_multiplier=.001
fric=.5 # higher = less friction. Used for wall-bounce only.
#permanent durations
current_fb_dur=5
iti_dur=.25
#fixation
fixation=psychopy.visual.Circle(win=win,pos=(0,0),color='white',radius=.002,edges=12)
#%% Block Start
for block in range(numblocks):
    blocktext=psychopy.visual.TextStim(win=win,
                        name='text',text='block'+str(block),pos=(.5,.46),
                        color='white',height=.015)
    blocktext.draw()
    core.wait(1)
    #%% ADD A "START BLOCK" INPUT!!!
    #%% Trial Start
    for trial in range(numtrialsperblock):
        #loop stuff
        frame_all=0
        frame_track=0
        # start stimulus physics
        c1_xpos=(random.randint(0,1000)/1000)*(poslmt-startpos_buf)
        c1_ypos=(random.randint(0,1000)/1000)*(poslmt-startpos_buf)
        c1_xacc=0*acc_multiplier
        c1_yacc=0*acc_multiplier
        # stimulus acceleration data
        c1_xacc_data=[0]
        c1_yacc_data=[0]
        c1_racc_data=[0]
        c1_deg_data=[0]
        # trail/fps data
        trail=[0]
        current_fps=[0.0]
        # durations/timing/randomized times
        current_cue_dur=random.randint(500,1500)/1000
        current_track_dur=random.randint(5000,15000)/1000 #5000-15000 is ok
        current_ri_dur=1
        # vwm reset conditions
        start_reset=random.randint(0,1)
        start_reset_time=random.randint(2000,(current_track_dur*1000)-2000)/1000
        reset_done=0
        #colors
        start_color=(1,1,1)
        reset_color=(1,1,0)
        # mouse tracking per trial
        mouse.clickReset()
        stim_pos=[[0.0,0.0]]
        mouse_pos=[[0.0,0.0]]
        #accuracy tracking
        current_acc_x=[0.0]
        current_acc_y=[0.0]
        if frame_prediction==1:
            acc_rad=[0.0]
            # Unidimensional Accuracy Descriptive Statistics
            acc_rad_mean=[0.0]
            acc_rad_error=[0.0]
            acc_rad_sqerror=[0.0]
            acc_rad_ss=[0.0]
            acc_rad_sd=[0.0]
            p_acc_rad_mean=[0.0]
            p_acc_rad_sd=[0.0]
        #%% INTER TRIAL INTERVAL (IVI)
        core.wait(iti_dur)
        #%% Cue
        # clock start
        trialClock=core.Clock()
        time_cue_start=trialClock.getTime()
        mouse.setPos([c1_xpos,c1_ypos])
        while 1:
            time_cue=trialClock.getTime()
            if time_cue>=current_cue_dur:
                time_cue_end=time_cue
                break
            #fixation
            tgtcircle=psychopy.visual.Circle(win=win,pos=(c1_xpos,c1_ypos),color='white',radius=c1_rad,edges=14)
            text=psychopy.visual.TextStim(win=win,
                        name='text',text='trial'+str(trial),pos=(.5,.46),
                        color='white',height=.015)
            # moving stims
            mousecircle=psychopy.visual.Circle(win=win,pos=(mouse.getPos()[0],mouse.getPos()[1]),color='grey',radius=c1_rad/3,edges=14)
            # then draw all stimuli
            tgtcircle.draw()
            mousecircle.draw()
            text.draw()
            fixation.draw()
            # then flip your window
            win.flip()
        #%% Track
        while 1:
            time_track=trialClock.getTime()
            # framerate set here... time%.004<.001
            if time_track%time_per_frame<time_per_frame_precision:
                # frame_track
                frame_track+=1
                # frame_track conditions
                # mouse/stim pos track
                stim_pos.append([c1_xpos,c1_ypos])
                mouse_pos.append([mouse.getPos()[0],mouse.getPos()[1]])
                # accuracy calculations
                current_acc_x.append(mouse_pos[frame_track][0]-stim_pos[frame_track][0])
                current_acc_y.append(mouse_pos[frame_track][1]-stim_pos[frame_track][1])
                # modulate acceleration towards center by distance from center
                c1_xacc=c1_xacc+(random.randint(-acclmt,acclmt)*acc_multiplier)+(-c1_xpos*opp_acc_multiplier)
                c1_yacc=c1_yacc+(random.randint(-acclmt,acclmt)*acc_multiplier)+(-c1_ypos*opp_acc_multiplier)
                # speedlimit
                if c1_xacc>spdlmt:
                    c1_xacc=spdlmt
                elif c1_xacc<-spdlmt:
                    c1_xacc=-spdlmt
                if c1_yacc>spdlmt:
                    c1_yacc=spdlmt
                elif c1_yacc<-spdlmt:
                    c1_yacc=-spdlmt
                c1_xpos=c1_xpos+c1_xacc
                c1_ypos=c1_ypos+c1_yacc
                # poslimit
                if c1_xpos>poslmt:
                    c1_xpos=poslmt
                    c1_xacc=-c1_xacc*fric
                elif c1_xpos<-poslmt:
                    c1_xpos=-poslmt
                    c1_xacc=-c1_xacc*fric
                if c1_ypos>poslmt:
                    c1_ypos=poslmt
                    c1_yacc=-c1_yacc*fric
                elif c1_ypos<-poslmt:
                    c1_ypos=-poslmt
                    c1_yacc=-c1_yacc*fric
                # stimulus acceleration data
                c1_xacc_data.append(c1_xacc)
                c1_yacc_data.append(c1_yacc)
                c1_racc_data.append(np.sqrt(c1_xacc_data[frame_track]**2+c1_yacc_data[frame_track]**2))
                # stimulus direction data
                c1_deg_data.append(np.arcsin((stim_pos[frame_track][1]-stim_pos[frame_track-1][1])/(stim_pos[frame_track][0]-stim_pos[frame_track-1][0])))
                if frame_track==1:
                    time_track_start=time_track
                # end trial if >current_track_dur
                if time_track>=current_track_dur:
                    frame_track_end=frame_track
                    time_track_end=time_track
                    time_track_dur=time_track_end-time_track_start
                    break
                # maybe start by making stimulus objects (e.g. myPic = visual.ImageStim(...))
                if debug==1:
                    text=psychopy.visual.TextStim(win=win,
                        name='text',text='trial'+str(trial),pos=(.5,.46),
                        color='white',height=.015)
                # draw the trail
                if trail_visual==1:
                    trail.append(psychopy.visual.Circle(win=win,pos=(c1_xpos,c1_ypos),color=[.1+round(c1_racc_data[frame_track]*100,1),0,0],colorSpace='rgb',radius=c1_rad/5,edges=3))
                    for current_frame in range(frame_track):
                        if current_frame%3 == 0:
                            trail[current_frame+1].draw()
                # position text
                if debug==1:
                    text_stim_pos=psychopy.visual.TextStim(win=win,
                        name='text',text='stim_pos'+str(stim_pos[frame_track]),pos=(.5,.34),
                        color='white',height=.015)
                    text_mouse_pos=psychopy.visual.TextStim(win=win,
                        name='text',text='mouse_pos'+str(mouse_pos[frame_track]),pos=(.5,.30),
                        color='white',height=.015)
                    text_time=psychopy.visual.TextStim(win=win,
                        name='text',text='time'+str(time_track-time_track_start),pos=(.5,.40),
                        color='white',height=.015)
                # accel/reset condition
                if reset_done==0 and start_reset==1 and current_track_dur>=start_reset_time:
                    #c1_xacc=0*acc_multiplier
                    #c1_yacc=0*acc_multiplier
                    reset_time=trialClock.getTime()
                    reset_done=1
                # fps & trial details text
                if reset_done==0:
                    # moving stims
                    tgtcircle=psychopy.visual.Circle(win=win,pos=(c1_xpos,c1_ypos),color=start_color,colorSpace='rgb',radius=c1_rad,edges=14)
                    mousecircle=psychopy.visual.Circle(win=win,pos=(mouse.getPos()[0],mouse.getPos()[1]),color=(.8,.8,.8),colorSpace='rgb',radius=c1_rad/3,edges=14)
                elif reset_done==1:
                    # moving stims
                    tgtcircle=psychopy.visual.Circle(win=win,pos=(c1_xpos,c1_ypos),color=reset_color,colorSpace='rgb',radius=c1_rad,edges=14)
                    mousecircle=psychopy.visual.Circle(win=win,pos=(mouse.getPos()[0],mouse.getPos()[1]),color=(.8,.8,.8),colorSpace='rgb',radius=c1_rad/3,edges=14)
                # fps & trial details text
                current_fps.append(round(frame_track/time_track,2))
                if debug_fps==1:
                    text_fps=psychopy.visual.TextStim(win=win,
                        name='text',text='fps'+str(current_fps[frame_track]),pos=(.5,.42),
                        color='white',height=.015)
                    text_frame=psychopy.visual.TextStim(win=win,
                        name='text',text='frame'+str(frame_track),pos=(.5,.38),
                        color='white',height=.015)
                if frame_prediction==1:
                    if frame_track==1:
                        # Collapse data to 1-dimensional (i.e., radius from target).
                        acc_rad.append(np.sqrt(current_acc_x[frame_track]**2+current_acc_y[frame_track]**2))
                        # Unidimensional Accuracy Descriptive Statistics
                        acc_rad_mean.append(np.mean(acc_rad[frame_track]))
                        acc_rad_error.append(0.0)
                        acc_rad_sqerror.append(0.0)
                        acc_rad_ss.append(0.0)
                        acc_rad_sd.append(0.0)
                        p_acc_rad_mean.append(0.0)
                        p_acc_rad_sd.append(0.0)
                    if frame_track>1:
                        # Collapse data to 1-dimensional (i.e., radius from target).
                        acc_rad.append(np.sqrt(current_acc_x[frame_track]**2+current_acc_y[frame_track]**2))
                        # Unidimensional Accuracy Descriptive Statistics
                        acc_rad_mean.append(np.mean(acc_rad[frame_track]))
                        acc_rad_error.append(acc_rad_mean[frame_track]-acc_rad[1::])
                        acc_rad_sqerror.append(acc_rad_error[frame_track]**2)
                        acc_rad_ss.append(sum(acc_rad_sqerror[frame_track]))
                        acc_rad_sd.append(acc_rad_ss[frame_track]/len(acc_rad[1::])-1)
                        p_acc_rad_mean.append(np.mean(acc_rad_mean[1::]))
                        p_acc_rad_sd.append(np.mean(acc_rad_sd[1::]))
                        if frame_prediction_visual==1:
                            predictcircle=psychopy.visual.Circle(win=win,pos=(c1_xpos,c1_ypos),color='blue',radius=p_acc_rad_mean[frame_track],edges=14)
                            stim_dir=psychopy.visual.Circle(win=win,pos=(c1_xpos+c1_xacc,c1_ypos+c1_yacc),color='white',radius=c1_rad/3,edges=3)
                if debug==1:
                    text.draw()
                    text_stim_pos.draw()
                    text_mouse_pos.draw()
                if debug_fps==1:
                    text_fps.draw()
                    text_frame.draw()
                    text_time.draw()
                # then draw all stimuli
                if frame_prediction==1 and frame_prediction_visual==1 and frame_track>1:
                    predictcircle.draw()
                tgtcircle.draw()
                mousecircle.draw()
                fixation.draw()
                # then flip your window
                win.flip()
        #%% By-trial Analysis
        # To access position data per trial:
        # d_xxx_pos[block][trial][timepoint]
        time_analysis_start=trialClock.getTime()
        trial_fps=frame_track_end/time_track_dur
        trial_tpf=time_track_dur/frame_track_end #time per frame (tpf)
        if trial_analysis==1:
            # Collapse data to 1-dimensional (i.e., radius from target).
            for current_acc in range(len(current_acc_x)):
                if current_acc==0:
                    acc_rad=[np.sqrt(current_acc_x[current_acc]**2+current_acc_y[current_acc]**2)]
                else:
                    acc_rad.append(np.sqrt(current_acc_x[current_acc]**2+current_acc_y[current_acc]**2))
            # Unidimensional Accuracy Descriptive Statistics
            acc_rad_mean=np.mean(acc_rad[1::])
            acc_rad_error=acc_rad_mean-acc_rad[1::]
            acc_rad_sqerror=acc_rad_error**2
            acc_rad_ss=sum(acc_rad_sqerror)
            acc_rad_sd=acc_rad_ss/len(acc_rad[1::])-1
            # Plot a 2d 'heatmap' of accuracy.
            for current_pix in range(len(acc_rad)):
                if current_pix==0:
                    heatmap=[psychopy.visual.Circle(win=win,pos=(current_acc_x[current_pix],current_acc_y[current_pix]),color=current_pix/len(acc_rad),colorSpace='rgb',radius=.001,edges=4)]
                else:
                    heatmap.append(psychopy.visual.Circle(win=win,pos=(current_acc_x[current_pix],current_acc_y[current_pix]),color=current_pix/len(acc_rad),colorSpace='rgb',radius=.001,edges=4))
            heatmap_mean=psychopy.visual.Circle(win=win,pos=(np.mean(current_acc_x),np.mean(current_acc_y)),color=(0,1,0),colorSpace='rgb',radius=.002,edges=4)
            #%% Fourier transform of mouse-stim accuracy throughout trial.
            fourier_freqs=np.arange(fourier_min_freq,fourier_max_freq,fourier_freq_res) #frequency in seconds of one complete fourier cycle.
            fourier_magnifier=40
            for freq in range(len(fourier_freqs)):
                if freq==0:
                    fourier_fpf=[trial_fps*fourier_freqs[freq]] #frames per fourier_freq
                    fourier_dpf=[360/fourier_fpf[freq]] #degrees per frame on 360 circular space
                else:
                    fourier_fpf.append(trial_fps*fourier_freqs[freq])
                    fourier_dpf.append(360/fourier_fpf[freq])
                # convert acc_rad into point along fourier circle.
                for acc in range(len(acc_rad)):
                    if acc==0:
                        fourier_acc_rad_x=[np.cos(fourier_dpf[freq]*acc)*acc_rad[acc]]
                        fourier_acc_rad_y=[np.sin(fourier_dpf[freq]*acc)*acc_rad[acc]]
                        fourier_acc_rad_rad=[np.sqrt(fourier_acc_rad_x[acc]**2+fourier_acc_rad_y[acc]**2)]
                        if fourier_freqs[freq]==4.0 and trial_analysis_visual==1:
                            fourier_graph=[psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=(0,1,1),colorSpace='rgb',radius=.001,edges=4)]
                        elif fourier_freqs[freq]==8.0 and trial_analysis_visual==1:
                            fourier_graph=[psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=(1,1,0),colorSpace='rgb',radius=.001,edges=4)]
                        elif trial_analysis_visual==1:
                            fourier_graph=[psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=freq/len(fourier_freqs),colorSpace='rgb',radius=.001,edges=4)]
                    else:
                        fourier_acc_rad_x.append(np.cos(fourier_dpf[freq]*acc)*acc_rad[acc])
                        fourier_acc_rad_y.append(np.sin(fourier_dpf[freq]*acc)*acc_rad[acc])
                        fourier_acc_rad_rad.append(np.sqrt(fourier_acc_rad_x[acc]**2+fourier_acc_rad_y[acc]**2))
                        if fourier_freqs[freq]==4.0 and trial_analysis_visual==1:
                            fourier_graph.append(psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=(0,1,1),colorSpace='rgb',radius=.001,edges=4))
                        elif fourier_freqs[freq]==8.0 and trial_analysis_visual==1:
                            fourier_graph.append(psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=(1,1,0),colorSpace='rgb',radius=.001,edges=4))
                        elif trial_analysis_visual==1:
                            fourier_graph.append(psychopy.visual.Circle(win=win,pos=(fourier_acc_rad_x[acc]*fourier_magnifier,fourier_acc_rad_y[acc]*fourier_magnifier),color=freq/len(fourier_freqs),colorSpace='rgb',radius=.001,edges=4))
                if freq==0:
                    #get fourier center of mass (CoM)
                    if trial_analysis_visual==1:
                        fourier_graph_freq=[fourier_graph]
                    fourier_com_x=[np.mean(fourier_acc_rad_x)]
                    fourier_com_y=[np.mean(fourier_acc_rad_y)]
                    fourier_com_rad=[np.sqrt(fourier_com_x[freq]**2+fourier_com_y[freq]**2)]
                    # Fourier CoM Descriptive Statistics
                    fourier_com_rad_mean=[fourier_com_rad[freq]]
                    fourier_com_rad_error=[fourier_com_rad_mean[freq]-fourier_acc_rad_rad[1::]]
                    fourier_com_rad_sqerror=[fourier_com_rad_error[freq]**2]
                    fourier_com_rad_ss=[sum(fourier_com_rad_sqerror[freq])]
                    fourier_com_rad_sd=[fourier_com_rad_ss[freq]/(len(acc_rad)-1)] #not sure about these.
                    # visual Fourier com
                    if fourier_freqs[freq]==4.0 and trial_analysis_visual==1:
                        fourier_com_dot=[psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.003,edges=4)]
                        fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.002,edges=4)]
                        #fourier_com_rad_sd_pos_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(0,0,1),colorSpace='rgb',radius=.001,edges=4)]
                        #fourier_com_rad_sd_neg_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(0,0,1),colorSpace='rgb',radius=.001,edges=4)]
                    elif fourier_freqs[freq]==8.0 and trial_analysis_visual==1:
                        fourier_com_dot=[psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.003,edges=4)]
                        fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.002,edges=4)]
                        #fourier_com_rad_sd_pos_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(1,0,0),colorSpace='rgb',radius=.001,edges=4)]
                        #fourier_com_rad_sd_neg_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(1,0,0),colorSpace='rgb',radius=.001,edges=4)]
                    elif trial_analysis_visual==1:
                        fourier_com_dot=[psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.002,edges=4)]
                        fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(1,1,1),colorSpace='rgb',radius=.001,edges=4)]
                        #fourier_com_rad_sd_pos_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.001,edges=4)]
                        #fourier_com_rad_sd_neg_graph=[psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.001,edges=4)]
                else:
                    #get fourier center of mass (com)
                    if trial_analysis_visual==1:
                        fourier_graph_freq.append(fourier_graph)
                    fourier_com_x.append(np.mean(fourier_acc_rad_x))
                    fourier_com_y.append(np.mean(fourier_acc_rad_y))
                    fourier_com_rad.append(np.sqrt(fourier_com_x[freq]**2+fourier_com_y[freq]**2))
                    # Fourier CoM Descriptive Statistics
                    fourier_com_rad_mean.append(fourier_com_rad[freq])
                    fourier_com_rad_error.append(fourier_com_rad_mean[freq]-fourier_acc_rad_rad[1::])
                    fourier_com_rad_sqerror.append(fourier_com_rad_error[freq]**2)
                    fourier_com_rad_ss.append(sum(fourier_com_rad_sqerror[freq]))
                    fourier_com_rad_sd.append(fourier_com_rad_ss[freq]/(len(acc_rad)-1))
                    #visual Fourier com
                    if fourier_freqs[freq]==4.0 and trial_analysis_visual==1:
                        fourier_com_dot.append(psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.003,edges=4))
                        fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.002,edges=4))
                        #fourier_com_rad_sd_pos_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(0,0,1),colorSpace='rgb',radius=.001,edges=4))
                        #fourier_com_rad_sd_neg_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(0,0,1),colorSpace='rgb',radius=.001,edges=4))
                    elif fourier_freqs[freq]==8.0 and trial_analysis_visual==1:
                        fourier_com_dot.append(psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.003,edges=4))
                        fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.002,edges=4))
                        #fourier_com_rad_sd_pos_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(1,0,0),colorSpace='rgb',radius=.001,edges=4))
                        #fourier_com_rad_sd_neg_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(1,0,0),colorSpace='rgb',radius=.001,edges=4))
                    elif trial_analysis_visual==1:
                        fourier_com_dot.append(psychopy.visual.Circle(win=win,pos=(fourier_com_x[freq]*fourier_magnifier,fourier_com_y[freq]*fourier_magnifier),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.002,edges=4))
                        fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,fourier_com_rad[freq]*fourier_magnifier),color=(1,1,1),colorSpace='rgb',radius=.001,edges=4))
                        #fourier_com_rad_sd_pos_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]+fourier_com_rad_sd[freq]),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.001,edges=4))
                        #fourier_com_rad_sd_neg_graph.append(psychopy.visual.Circle(win=win,pos=(.2+freq*.01,fourier_com_rad_mean[freq]-fourier_com_rad_sd[freq]),color=(0,freq/len(fourier_freqs),0),colorSpace='rgb',radius=.001,edges=4))
        #%% RI
        # RI - time taken to do the trial analyses.
        time_analysis_dur=trialClock.getTime()-time_analysis_start
        # then draw all stimuli
        if debug==1:
            text.draw()
            text_stim_pos.draw()
            text_mouse_pos.draw()
        if debug_fps==1:
            text_fps.draw()
            text_frame.draw()
            text_time.draw()
        fixation.draw()
        # then flip your window
        win.flip()
        core.wait(current_ri_dur-time_analysis_dur)
        #%% By-trial analysis Feedback
        # Heatmap
        if trial_analysis==1 and trial_analysis_visual==1:
            if debug==1:
                text_heatmap=psychopy.visual.TextStim(win=win,
                    name='text',text='trial accuracy t-collapsed heatmap',pos=(.7,.40),
                    color='white',height=.015)
                text_heatmap_mean=psychopy.visual.TextStim(win=win,
                    name='text',text='white:accuracy, green:mean accuracy',pos=(.7,.45),
                    color='white',height=.015)
                text_fb=psychopy.visual.TextStim(win=win,
                        name='text',text=str(round(np.mean(acc_rad),2)*100),pos=(.7,.35),
                        color='white',height=.015)
                #draw heatmap
                text_fb.draw()
                text_heatmap.draw()
                text_heatmap_mean.draw()
            fixation.draw()
            for current_pix in range(len(heatmap)):
                heatmap[current_pix].draw()
            heatmap_mean.draw()
            # then flip your window
            win.flip()
            core.wait(current_fb_dur/2)
            if debug==1:
                # Fourier transform
                text_fourier=psychopy.visual.TextStim(win=win,
                    name='text',text='trial accuracy fourier transform',pos=(.7,.45),
                    color='white',height=.015)
                text_fourier_desc=psychopy.visual.TextStim(win=win,
                    name='text',text='cyan:4hz blue:com, pink:8hz red:com, white:other hz green:com',pos=(.7,.4),
                    color='white',height=.015)
                #draw Fourier
                text_fourier.draw()
                text_fourier_desc.draw()
            fixation.draw()
            for current_freq in range(len(fourier_graph_freq)):
                for current_pix in range(len(fourier_graph)):
                    fourier_graph_freq[current_freq][current_pix].draw()
                    fourier_graph_freq[freq==4.0][current_pix].draw()
                    fourier_graph_freq[freq==8.0][current_pix].draw()
                fourier_com_dot[current_freq].draw()
            fourier_com_dot[freq==4.0].draw()
            fourier_com_dot[freq==8.0].draw()
            # then flip your window
            win.flip()
            core.wait(current_fb_dur*1.5)
            if debug==1:
                # Fourier com graph
                text_fourier_com=psychopy.visual.TextStim(win=win,
                    name='text',text='fourier: center of mass amplitude by frequency',pos=(.7,.45),
                    color='white',height=.015)
                text_fourier_com_desc=psychopy.visual.TextStim(win=win,
                    name='text',text='blue:4hz com, red:8hz com, white:other hz com',pos=(.7,.4),
                    color='white',height=.015)
                # draw fourier com graphs
                text_fourier_com.draw()
                text_fourier_com_desc.draw()
            fixation.draw()
            for current_freq in range(len(fourier_graph_freq)):
                fourier_com_rad_graph[current_freq].draw()
            fourier_com_rad_graph[freq==4.0].draw()
            fourier_com_rad_graph[freq==8.0].draw()
            # then flip your window
            win.flip()
            core.wait(current_fb_dur*1.5)
        #%% Playback (For debugging)
        ## Performance View
        if playback==1:
            performanceClock=core.Clock()
            frame_2=-1
            while 1:
                time_track_2=performanceClock.getTime()
                if time_track_2%trial_tpf<time_per_frame_precision:
                    frame_2+=1
                    if frame_2>=frame_track_end:
                        break
                    # stim pos and mouse pos
                    tgtcircle_2=psychopy.visual.Circle(win=win,pos=(stim_pos[frame_2][0],stim_pos[frame_2][1]),color='white',radius=c1_rad,edges=14)
                    mousecircle_2=psychopy.visual.Circle(win=win,pos=(mouse_pos[frame_2][0],mouse_pos[frame_2][1]),color='grey',radius=c1_rad/3,edges=3)
                    # fps & trial details text
                    text_frame=psychopy.visual.TextStim(win=win,
                        name='text',text='frame'+str(frame_2),pos=(.5,.38),
                        color='white',height=.015)
                    text_stim_pos=psychopy.visual.TextStim(win=win,
                        name='text',text='stim_pos'+str(stim_pos[frame_2]),pos=(.5,.34),
                        color='white',height=.015)
                    text_mouse_pos=psychopy.visual.TextStim(win=win,
                        name='text',text='stim_pos'+str(mouse_pos[frame_2]),pos=(.5,.30),
                        color='white',height=.015)
                    text_time=psychopy.visual.TextStim(win=win,
                        name='text',text='time'+str(time_track_dur*frame_2/frame_track_end),pos=(.5,.40),
                        color='white',height=.015)
                    # then draw all stimuli
                    text.draw()
                    tgtcircle_2.draw()
                    mousecircle_2.draw()
                    text_frame.draw()
                    text_stim_pos.draw()
                    text_mouse_pos.draw()
                    fixation.draw()
                    # then flip your window
                    win.flip()
        #%% Save data!
        # To access position data per trial:
        # xxxx_pos_data[block][trial][timepoint]
        if trial==0 and block==0:
            #trial data
            d_block=[block]
            d_trial=[trial]
            d_cue_dur=[current_cue_dur]
            d_track_dur=[current_track_dur]
            d_time_cue_start=[time_cue_start]
            d_time_cue_end=[time_cue_end]
            d_time_track_start=[time_track_start]
            d_time_track_end=[time_track_end]
            d_time_track_dur=[time_track_dur]
            d_frame_track_end=[frame_track_end]
            d_current_fps=[current_fps]
            d_start_reset=[start_reset]
            #raw positions
            d_stim_pos=[stim_pos]
            d_mouse_pos=[mouse_pos]
            #raw acceleration and degrees data
            d_c1_xacc_data=[c1_xacc_data]
            d_c1_yacc_data=[c1_yacc_data]
            d_c1_racc_data=[c1_racc_data]
            d_c1_deg_data=[c1_deg_data]
            if trial_analysis==1:
                #heatmap
                d_acc_x=[current_acc_x]
                d_acc_y=[current_acc_y]
                d_acc_rad=[acc_rad]
                #fourier
                d_fourier_acc_rad_x=[fourier_acc_rad_x]
                d_fourier_acc_rad_y=[fourier_acc_rad_y]
                d_fourier_com_x=[fourier_com_x]
                d_fourier_com_y=[fourier_com_y]
                d_fourier_com_rad=[fourier_com_rad]
                #trial statistics
                d_acc_rad_mean=[acc_rad_mean]
                d_acc_rad_sd=[acc_rad_sd]
                d_fourier_com_rad_mean=[fourier_com_rad_mean]
                d_fourier_com_rad_sd=[fourier_com_rad_sd]
        else:
            #trial data
            d_block.append(block)
            d_trial.append(trial)
            d_cue_dur.append(current_cue_dur)
            d_track_dur.append(current_track_dur)
            d_time_cue_start.append(time_cue_start)
            d_time_cue_end.append(time_cue_end)
            d_time_track_start.append(time_track_start)
            d_time_track_end.append(time_track_end)
            d_time_track_dur.append(time_track_dur)
            d_frame_track_end.append(frame_track_end)
            d_current_fps.append(current_fps)
            d_start_reset.append(start_reset)
            #raw positions
            d_stim_pos.append(stim_pos)
            d_mouse_pos.append(mouse_pos)
            #raw acceleration and degrees data
            d_c1_xacc_data.append(c1_xacc_data)
            d_c1_yacc_data.append(c1_yacc_data)
            d_c1_racc_data.append(c1_racc_data)
            d_c1_deg_data.append(c1_deg_data)
            if trial_analysis==1:
                #heatmap
                d_acc_x.append(current_acc_x)
                d_acc_y.append(current_acc_y)
                d_acc_rad.append(acc_rad)
                #fourier
                d_fourier_acc_rad_x.append(fourier_acc_rad_x)
                d_fourier_acc_rad_y.append(fourier_acc_rad_y)
                d_fourier_com_x.append(fourier_com_x)
                d_fourier_com_y.append(fourier_com_y)
                d_fourier_com_rad.append(fourier_com_rad)
                #trial statistics
                d_acc_rad_mean.append(acc_rad_mean)
                d_acc_rad_sd.append(acc_rad_sd)
                d_fourier_com_rad_mean.append(fourier_com_rad_mean)
                d_fourier_com_rad_sd.append(fourier_com_rad_sd)
        #%% By-Trial Prediction! Predict accuracy and fouriers.
        if trial_analysis & trial_prediction==1:
            time_prediction_start=trialClock.getTime()
            if trial==0:
                p_acc_rad_mean=d_acc_rad_mean[trial]
                p_acc_rad_sd=d_acc_rad_sd[trial]
                p_fourier_com_rad_mean=np.zeros(len(fourier_freqs))
                p_fourier_com_rad_sd=np.zeros(len(fourier_freqs))
                pre_fourier_com_rad_mean=np.zeros(len(fourier_freqs))
                pre_fourier_com_rad_sd=np.zeros(len(fourier_freqs))
                for current_freq in range(len(fourier_freqs)):
                    pre_fourier_com_rad_mean[current_freq]=d_fourier_com_rad_mean[0][current_freq]
                    pre_fourier_com_rad_sd[current_freq]=d_fourier_com_rad_sd[0][current_freq]
            else:
                p_acc_rad_mean=sum(d_acc_rad_mean)/(trial+1)
                p_acc_rad_sd=sum(d_acc_rad_sd)/(trial+1)
                for current_trial in range(trial):
                    for current_freq in range(len(fourier_freqs)):
                        pre_fourier_com_rad_mean[current_freq]=pre_fourier_com_rad_mean[current_freq]+d_fourier_com_rad_mean[current_trial][current_freq]
                        pre_fourier_com_rad_sd[current_freq]=pre_fourier_com_rad_sd[current_freq]+d_fourier_com_rad_sd[current_trial][current_freq]
                for current_freq in range(len(fourier_freqs)):
                    p_fourier_com_rad_mean[current_freq]=pre_fourier_com_rad_mean[current_freq]/(trial+1)
                    p_fourier_com_rad_sd[current_freq]=pre_fourier_com_rad_sd[current_freq]/(trial+1)
            time_prediction_end=trialClock.getTime()-time_prediction_start
        #%% Save data!
            if trial==0 and block==0:
                d_time_prediction_end=[time_prediction_end]
            else:
                d_time_prediction_end.append(time_prediction_end)
        #%% By-trial prediction Feedback
        # P_Fourier com graph
        if trial_prediction==1 and trial_prediction_visual==1:
            if debug==1:
                # Fourier com graph
                text_fourier_com=psychopy.visual.TextStim(win=win,
                    name='text',text='fourier: center of mass amplitude by frequency',pos=(.7,.45),
                    color='white',height=.015)
                text_fourier_com_desc=psychopy.visual.TextStim(win=win,
                    name='text',text='blue:4hz com, red:8hz com, white:other hz com',pos=(.7,.4),
                    color='white',height=.015)
                # draw fourier com graphs
                text_fourier_com.draw()
                text_fourier_com_desc.draw()
            fixation.draw()
            for current_freq in range(len(fourier_freqs)):
                if current_freq==0:
                    if fourier_freqs[current_freq]==4.0:
                        p_fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.002,edges=4)]
                    elif fourier_freqs[current_freq]==8.0:
                        p_fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.002,edges=4)]
                    else:
                        p_fourier_com_rad_graph=[psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(1,1,1),colorSpace='rgb',radius=.001,edges=4)]
                else:
                    if fourier_freqs[current_freq]==4.0:
                        p_fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+current_freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(0,0,1),colorSpace='rgb',radius=.002,edges=4))
                    elif fourier_freqs[current_freq]==8.0:
                        p_fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+current_freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(1,0,0),colorSpace='rgb',radius=.002,edges=4))
                    else:
                        p_fourier_com_rad_graph.append(psychopy.visual.Circle(win=win,pos=(-.1*fourier_magnifier*.1+current_freq*fourier_freq_res*.01*fourier_magnifier*.1,p_fourier_com_rad_mean[current_freq]*fourier_magnifier),color=(1,1,1),colorSpace='rgb',radius=.001,edges=4))
                p_fourier_com_rad_graph[current_freq].draw()
            p_fourier_com_rad_graph[freq==4.0].draw()
            p_fourier_com_rad_graph[freq==8.0].draw()
            # then flip your window
            win.flip()
            core.wait(current_fb_dur*1.5)
            
#%% Save All Data!
d_block = np.asarray(d_block)
d_trial = np.asarray(d_trial)
d_cue_dur = np.asarray(d_cue_dur)
d_track_dur = np.asarray(d_track_dur)
d_time_track_dur = np.asarray(d_time_track_dur)
d_start_reset = np.asarray(d_start_reset)
d_stim_pos = np.asarray(d_stim_pos)
d_mouse_pos = np.asarray(d_mouse_pos)
d_c1_xacc_data = np.asarray(d_c1_xacc_data)
d_c1_yacc_data = np.asarray(d_c1_yacc_data)
d_c1_racc_data = np.asarray(d_c1_racc_data)
d_c1_deg_data = np.asarray(d_c1_deg_data)
d_acc_x = np.asarray(d_acc_x)
d_acc_y = np.asarray(d_acc_y)
d_acc_rad = np.asarray(d_acc_rad)

np.savetxt("d_block.csv", d_block, delimiter=",")
np.savetxt("d_trial.csv", d_trial, delimiter=",")
np.savetxt("d_cue_dur.csv", d_cue_dur, delimiter=",")
np.savetxt("d_track_dur.csv", d_track_dur, delimiter=",")
np.savetxt("d_time_track_dur.csv", d_time_track_dur, delimiter=",")
np.savetxt("d_start_reset.csv", d_start_reset, delimiter=",")
np.savetxt("d_stim_pos.csv", d_stim_pos, delimiter=",")
np.savetxt("d_mouse_pos.csv", d_mouse_pos, delimiter=",")
np.savetxt("d_c1_xacc_data.csv", d_c1_xacc_data, delimiter=",")
np.savetxt("d_c1_yacc_data.csv", d_c1_yacc_data, delimiter=",")
np.savetxt("d_c1_racc_data.csv", d_c1_racc_data, delimiter=",")
np.savetxt("d_c1_deg_data.csv", d_c1_deg_data, delimiter=",")
np.savetxt("d_acc_x.csv", d_acc_x, delimiter=",")
np.savetxt("d_acc_y.csv", d_acc_y, delimiter=",")
np.savetxt("d_acc_rad.csv", d_acc_rad, delimiter=",")

#%% Required clean up
# this cell will make sure that your window displays for a while and then 
# closes properly
core.wait(1)
win.close()
#%% To Do List
# After tracking phase onset, a sudden change in mouse acceleration AFTER stimulus
# starts moving AND after mouse is outside stimulus circle == RT response to stimulus
# moving start. Do we need stim to stay still at end?
## Machine learn mouse acceleration during tracking. If that algorithm can't
# predict tracking, then implied to be mouse accelerating due to object location reset,
# or lack of movement (no attention relayed to tracking).
## RT at beginning of trial serves as how much lag is expected if NO
# contribution of expectation of stimulus.
## Machine learn mouse movement during tracking (how to categorize?) as a function
# of acceleration of stimulus when timeback == 1 (acceleration avg of 1 frame) or more.
## In the end, I don't know what I am measuring but it will be cool.
#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import animation
from collections import defaultdict

dataaddr = r"/Raw-Data-Experiment/20240725/emulsion data_stoma with or without fuel/"
filenames=os.listdir(dataaddr)

# Example list of files
files = filenames

# Dictionary to hold categorized files
categorized_files = defaultdict(list)

for file in files:
    filename_without_ext = os.path.splitext(file)[0]
    parts = filename_without_ext.split('_')

    # Automatically extract the part containing `urea+dye`
    for part in parts:
        if 'mMurea+dye' in part:
            category = part
            break
    else:
        category = "Unknown"  # Handle cases where 'urea+dye' is not found

    # Add the full file name to the corresponding category
    categorized_files[category].append(file)specific_category = '10mMurea+dye'
if specific_category in categorized_files:
    print(f"Files in category '{specific_category}':")
    for f in categorized_files[specific_category]:
        print(f"  {f}")
else:
    print(f"No files found in category '{specific_category}'.")
# In[6]:


# Dictionary to hold categorized files
categorized_files = defaultdict(list)

# Categorizing files
for file in files:
    filename_without_ext = os.path.splitext(file)[0]
    parts = filename_without_ext.split('_')

    # Automatically extract the part containing `urea+dye`
    for part in parts:
        if 'mMurea+dye' in part:
            category = part
            break
    else:
        category = "Unknown"  # Handle cases where 'urea+dye' is not found

    # Add the full file name to the corresponding category
    categorized_files[category].append(file)


# Specify the specific category you want to process
specific_categories = ['0mMurea+dye', '10mMurea+dye', '25mMurea+dye', '100mMurea+dye', '500mMurea+dye']

Movieaddr = r"/Tracking-Experimental-Data/20240725/emulsion data_stoma with or without fuel/"
input_videos = []
for specific_category in specific_categories:
    for f in categorized_files[specific_category]:
        Moviefilename = os.path.join(Movieaddr, os.path.splitext(f)[0], 'trajectory.avi')
        input_videos.append(Moviefilename)
    
    # Output video path
    output_video_path = './Movie/combined_' + str(specific_category) + '.mp4'
    
    # Initialize VideoCapture for each video
    video_clips = []
    for video in input_videos:
        # Check if the video file exists before trying to open
        if not os.path.exists(video):
            print(f"Error: Video file {video} not found.")
        else:
            print(f"Opening video {video}")
            video_clip = cv2.VideoCapture(video)
            if not video_clip.isOpened():
                print(f"Error opening video: {video}")
            else:
                video_clips.append(video_clip)
    num_videos = len(video_clips)
    
    # Calculate the grid size: square root of the number of videos
    grid_rows = math.ceil(math.sqrt(num_videos))  # Number of rows
    grid_cols = math.ceil(num_videos / grid_rows)  # Number of columns
    
    # Set the grid size dynamically
    grid_size = (grid_rows, grid_cols)
    
    # If no valid video clips were opened, exit early
    if len(video_clips) == 0:
        print("No valid video clips to process.")
    else:
        # Get video properties (assuming all videos have the same properties)
        frame_width = int(video_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_clips[0].get(cv2.CAP_PROP_FPS)
    
        # Create VideoWriter to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * grid_size[1], frame_height * grid_size[0]))
    
        # Loop through frames
        while True:
            frames = []
    
            # Read frames from each video clip
            for idx, clip in enumerate(video_clips):
                ret, frame = clip.read()
                if not ret:
                    frames.append(None)  # Append None if frame could not be read
                else:
                    # Get the folder path of the video
                    video_path = input_videos[idx]  # Get the current video path
                    
                    # Extract the folder name from the path
                    folder_name = os.path.basename(os.path.dirname(video_path))  # Extract folder name
                    
                    
                    parts = folder_name.split('_')  # Split by underscores
                    main_title = '_'.join(parts[1:3])  
                    
                   
                    sub_title = "_".join(parts[3:])  # Get the title for each specific movie
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    
                    cv2.putText(frame, main_title, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
                    
                    cv2.putText(frame, sub_title, (10, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
                    frames.append(frame)
    
            if all(frame is None for frame in frames):
                break  # End of all videos
    
            # Resize frames to match grid layout
            resized_frames = [cv2.resize(frame, (frame_width, frame_height)) if frame is not None else np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for frame in frames]
    
            # Ensure grid consistency by filling empty slots with black frames (if necessary)
            while len(resized_frames) < grid_size[0] * grid_size[1]:
                resized_frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))  # Add blank frames if needed
    
            # Create rows from the resized frames and stack them to form a grid
            rows = []
            for i in range(grid_size[0]):
                row_frames = resized_frames[i * grid_size[1]:(i + 1) * grid_size[1]]
                row = np.hstack(row_frames)  # Horizontally stack frames for each row
                rows.append(row)
            
            # Stack top and bottom rows vertically to create the final grid
            combined_frame = np.vstack(rows)
    
            # Write combined frame to output video
            out.write(combined_frame)
    
        # Release VideoCapture and VideoWriter objects
        for clip in video_clips:
            clip.release()
        out.release()
    
        print(f"Combined video saved to {output_video_path}")
       





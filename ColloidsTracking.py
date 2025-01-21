#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pickle
import pims
import trackpy as tp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
import cv2
import matplotlib.cm as cm
tp.quiet(suppress=True)
import nd2
import nd2reader
from scipy.optimize import curve_fit
import os
import cmasher as cmr
from scipy.spatial import KDTree




@pims.pipeline
def preprocess(frame, l_blur):
    avrg = np.average(frame)
    return cv2.GaussianBlur(abs(frame - avrg), (l_blur, l_blur), 0)

class System():
    def __init__(self,
                 savefilename=None,
                 datafilename=None,
                 savefiledirectory=None,
                 series_id=None,
                 T=None,
                 locator_params=None,
                 l_blur=None,
                 cluster_radius=None,
                 search_range=None,
                 memory=None,
                 min_stub_frame_length=None):
        self.savefilename = savefilename
        self.datafilename = datafilename
        self.savefiledirectory=savefiledirectory
        self.series_id = series_id
        self.T = T
        self.locator_params = locator_params
        self.cluster_radius = cluster_radius
        self.l_blur = l_blur
        self.search_range = search_range
        self.memory = memory
        self.min_stub_frame_length = min_stub_frame_length
        self.mpp = 0.16  # microns per pixel
        self.fps = 31.2  # frames per second
        print(self.savefiledirectory)
    def Get_Original_Frames(self):
        self.frames = nd2reader.Nd2(self.datafilename)[:self.T]
    def Get_Frames(self):
        
        reader= nd2reader.Nd2(self.datafilename)
        frames_original = [frame for frame in reader]
        frames = preprocess(frames_original, self.l_blur)
        frames = frames[:self.T]
        self.frames = frames
    def Find_Tracks(self):
        # Remove 'engine' if it's in the locator_params
        if 'engine' in self.locator_params:
            del self.locator_params['engine']
        
        # Now call tp.batch with the engine='python' argument
        self.features = tp.batch(self.frames, processes=1, engine='python', **self.locator_params)
        tracks = tp.link(self.features, search_range=self.search_range, memory=self.memory)
        #tp.plot_traj(tracks,superimpose=self.frames[0])
        self.tracks = tracks
        self.tracks_0 = self.tracks.copy()

    def _remove_clusters_kdtree(self, f, radius):
        to_keep = np.ones(len(f), dtype=bool)
        for frame_id, frame_data in f.groupby('frame'):
            coords = np.vstack((frame_data.x, frame_data.y)).T
            tree = KDTree(coords)

            # Set of points in a cluster this frame
            removed = set()
            for i in range(len(frame_data)):
                if i in removed:
                    continue  # Skip points already marked for removal
                indices = tree.query_ball_point(coords[i], r=radius)
                if len(indices) > 1:  # Cluster found
                    for j in indices:
                        removed.add(j)

            original_indices = frame_data.index
            to_keep[list(original_indices[list(removed)])] = False
        return f[to_keep].reset_index(drop=True)

    def Filter_Stubs(self):
        print('Before:', self.tracks['particle'].nunique())
        self.tracks = tp.filter_stubs(self.tracks, self.min_stub_frame_length)
        print('After:', self.tracks['particle'].nunique())

    def Remove_Drift(self):
        d = tp.compute_drift(self.tracks)
        self.tracks = tp.subtract_drift(self.tracks.copy(), d)

    def Calculate_MSD_quadratic(self):
        def model(t, v, D):
            return v**2 * t**2 + 4 * D * t

        lower_bounds = [0, 0]  # Both v and D should be greater than or equal to zero
        upper_bounds = [np.inf, np.inf]  # No upper limit
        bounds = (lower_bounds, upper_bounds)
        self.em = tp.emsd(self.tracks, self.mpp, self.fps)#, max_lagtime=self.min_stub_frame_length)
        x_data = self.em.index
        y_data = self.em
        # Fit the model to the data
        popt, pcov = curve_fit(model, x_data, y_data, bounds=bounds)
        self.v_opt, self.D_opt = popt
        perr = np.sqrt(np.diag(pcov))
        self.v_err, self.D_err = perr
        print(f"Optimized v: {self.v_opt} ± {self.v_err}")
        print(f"Optimized D: {self.D_opt} ± {self.D_err}")

    def Calculate_MSD_linear(self):
    # Define the model that includes both D and alpha
        def model(t, D, alpha):
            return D * t ** alpha
    
        # Bounds for D and alpha (you may want to fine-tune these)
        lower_bounds = [0, 0]  # D >= 0 and alpha >= 0
        upper_bounds = [np.inf, 2]  # No upper limit for D, alpha can be up to 2 (for normal diffusion and superdiffusion)
        bounds = (lower_bounds, upper_bounds)
    
        # Compute the ensemble MSD
        self.em = tp.emsd(self.tracks, self.mpp, self.fps)#, max_lagtime=self.min_stub_frame_length)
    
        # Get x_data (lag times) and y_data (MSD values)
        x_data = self.em.index
        y_data = self.em
    
        # Fit the model to the data
        popt, pcov = curve_fit(model, x_data, y_data, bounds=bounds)
    
        # Extract optimized parameters: D and alpha
        self.D_opt, self.alpha_opt = popt
    
        # Calculate the errors for D and alpha
        perr = np.sqrt(np.diag(pcov))
        self.D_err, self.alpha_err = perr
    
        # Print the results
        print(f"Optimized D: {self.D_opt} ± {self.D_err}")
        print(f"Optimized alpha: {self.alpha_opt} ± {self.alpha_err}")

    def Save(self):
        self.frames = None
        with open(self.savefilename, 'wb') as savefile:
            pickle.dump(self, savefile)

    @staticmethod
    def Load(savefilename):
        with open(savefilename, 'rb') as savefile:
            return pickle.load(savefile)


    


    def create_video(self):
        """
        Create a video of the particle tracks and save it as an AVI file with colors.
        """
        
        t1 = self.tracks
        images = self.frames  
    
        # Generate a colormap for the particles
        particle_ids = t1['particle'].unique()  # Get unique particle IDs
        num_particles = len(particle_ids)
        #cmap = plt.get_cmap('jet')  # Get 'jet' colormap
        cmap =cmr.guppy
    
        # Map each particle to a color
        colors = {pid: cmap(i / (num_particles - 1)) for i, pid in enumerate(particle_ids)}  # Normalize index
    
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(images[0], cmap='gray', origin='lower')  
        scatter = ax.scatter([], [], s=1, alpha=0.5)  
        ax.set_axis_off()
    
        # Update function for the frame rendering
        def update(frame_number):
            img.set_data(images[frame_number])  # Update the image data
            traj_frame = t1[t1['frame'] <= frame_number]
    
            x, y = traj_frame['x'], traj_frame['y']
            particle_ids = traj_frame['particle']
            scatter.set_offsets(np.column_stack((x, y)))
            scatter.set_color([colors[pid] for pid in particle_ids])  # Set color by particle ID
    
            # Draw trajectory lines for each particle
            for line in ax.lines[:]:
                line.remove()  # Remove each line explicitly
            for pid, color in colors.items():
                particle_traj = traj_frame[traj_frame['particle'] == pid]
                ax.plot(particle_traj['x'], particle_traj['y'], color=color, alpha=0.6)
            ax.set_axis_off()
    
        # Prepare the AVI writer
        output_path = os.path.join(self.savefiledirectory, "trajectory.avi")
        output_path_temp = os.path.join(self.savefiledirectory, "temp_frame.png")
        height, width = images[0].shape[:2]  # Assuming grayscale images
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=True)
    
        # Render and save each frame
        for frame_number in range(len(images)):
            update(frame_number)  # Update the plot for the current frame
    
            # Save the current frame as an image
            plt.savefig(output_path_temp, bbox_inches='tight', pad_inches=0)
            frame = cv2.imread(output_path_temp)  # Read as a color image
            frame_resized = cv2.resize(frame, (width, height))  # Ensure correct size
            out.write(frame_resized)  # Write the frame to the AVI file
    
        # Release resources
        out.release()
        os.remove(output_path_temp)  # Clean up the temporary file
        plt.close(fig)

        print(f"Video saved as 'trajectory.avi' at {output_path}.")


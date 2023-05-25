import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from enum import Enum
import warnings
import math

class probeArrayQueue:
    """
    A class that maintains a queue of 50 4x4 matrix
    """
    
    def __init__(self):
        self.queue = []
        self.max_size = 200
        
    def add_item(self, matrix):
        if len(self.queue) == self.max_size:
            self.queue.pop(0)
        self.queue.append(matrix)
        
    def get_num_element(self):
        return len(self.queue)
        
    def is_queue_full(self):
        return len(self.queue) == self.max_size
    
    def get_yy_zz_tt(self):
        yy = []
        zz = []
        tt = []
        for matrix in self.queue:
            yy.append(matrix[0][0])
            zz.append(matrix[0][1])
            tt.append(matrix[0][2])
        return yy, zz, tt


# test_params = {"amplitude_y" : 8, "amplitude_z" : 6, "omega": 2.132349, "phase" : 1.0, "offset" : 2.6945}

## Global parameters
class MotionTrackingPhase(Enum):
    PREDICTION = 1
    MOVING = 2

class MotionTracking:
    def __init__(self):
       self.phase = MotionTrackingPhase.PREDICTION
       pass

    ################## Change the Motion Tracking Phase #####################
    def change_motion_tracking_phase(self, current_phase):
        self.phase = MotionTrackingPhase.PREDICTION = current_phase
        pass
    
    ################## Get Current Phase of Motion Tracking #####################    
    def get_motion_tracking_phase(self):
        return self.phase
    
    ################## Preprocess Data #####################
    def preprocessing_data(self, probe_pose_0, probe_pose, registration_matrix):
        """
        Perform preprocessing on registration data to extract actual kidney position

        @param probe_pose_0: The initial probe pose at the momment reference frame is taken
        @param probe_pose: The current probe pose.
        @param registration_matrix: output matrix from registration.

        @return: New position of kidney (in probe_0 frame)
        """
        phase = self.phase.value
        probe_T_image = np.array([[0,0,-1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64) # the transformation of image frame w.r.t probe, ignore the translation

        if phase == 1 : # Prediction phase
            relative_probe_pose = np.identity(4) # probe is not moving yet
        elif phase == 2 : # Moving phase
            relative_probe_pose = np.dot(np.linalg.inv(probe_pose_0), probe_pose)
    
        probe_T_registration = np.dot(probe_T_image, registration_matrix)
        probe_T_kidney = np.dot(relative_probe_pose, probe_T_registration)
        p_y = probe_T_kidney[1][3]
        p_z = probe_T_kidney[2][3]
        return p_y, p_z



    ################## Fit the sinuoidal model #####################
    def fit_sin(self, tt, zz, yy): # passing np.array of t and position
        """
        Perform sinusoidal regression to generalize motion model

        @param tt: array of time data
        @param zz: array of kidney in z axis (probe frame)
        @param yy: array of kidney in y axis (probe frame)

        @return: the fitted motion model parrameters,  
        """
        # if (len(zz) < 50 or len(yy) < 50) :
        #     return
    
        ny = 1 # in probe frame, y is lateral move
        nz = 0 # in probe frame, z is up and down move

        '''Linear regresion to find the direction of the motion '''
        lin_regression_result = linregress(zz, yy)
        #print(f'---------------------- Linear Regression Result ------------')
        #print(f'R-value {lin_regression_result.rvalue:.3f}.')
        #print(f'Slope {lin_regression_result.slope:.3f}.')
    
        if (lin_regression_result.rvalue < 0.9) :
            warnings.warn("The trajectory is not really in a line")

        if (1/lin_regression_result.slope > 0.25) :
            nz = 1.0/math.sqrt(1.0  + math.pow(lin_regression_result.slope, 2))
            ny = math.sqrt(1.0 - nz**2)
            warnings.warn("The trajectory is not just going along the y axis")

        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
        popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        self.change_motion_tracking_phase = MotionTrackingPhase.MOVING
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "ny": ny, "nz" : nz, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

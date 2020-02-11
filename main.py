import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from tempfile import TemporaryFile
import re
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import errno
import os
import sys
import plotly.plotly as py

class ActivityRecognition:
    def __init__(self, ground_truth_dir_path=None, myo_data_dir_path=None):
        self.__data_dir_path = ground_truth_dir_path
        self.__plot = True
        self.__save = False
        self.__show_plot = False
        self.__log_enabled = False
        self.__save_files = False                        
        self.__ground_truth_file_list = []
        self.__figure_number = 0
        self.__plots_dir = 'plots'
        self.__raw_plots_dir_path = []
        self.__pca_plots_dir_path = []
        self.__orientation_axis = {"x" : 0, "y":1, "z" :2, "w" : 3}
        self.__accelerometer_axis = {"x" : 4, "y":5, "z" :6}
        self.__gyro_axis = {"x" : 7, "y":8, "z" :9}
    
        self.__all_imu_myo_fork_data = []
        self.__imu_myo_fork_data_userid = []
        self.__all_imu_myo_spoon_data = []
        self.__imu_myo_spoon_data_userid = []
        self.__imu_fork_eating = []
        self.__imu_fork_non_eating = []
        self.__imu_spoon_eating = []
        self.__imu_spoon_non_eating = []

        # MEAN
        self.__mean_eating_fork = []
        self.__mean_non_eating_fork = []
        self.__mean_eating_spoon= []
        self.__maen_non_eating_spoon = []

        # VAR
        self.__variance_eating_fork = []
        self.__variance_non_eating_fork = []
        self.__variance_eating_spoon= []
        self.__variance_non_eating_spoon = []

        # RMS
        self.__rms_eating_fork = []
        self.__rms_non_eating_fork = []
        self.__rms_eating_spoon= []
        self.__rms_non_eating_spoon = [] 

        # MAX
        self.__min_eating_fork = []
        self.__min_non_eating_fork = []
        self.__min_eating_spoon= []
        self.__min_non_eating_spoon = []

        # MIN
        self.__max_eating_fork = []
        self.__max_non_eating_fork = []
        self.__max_eating_spoon= []
        self.__max_non_eating_spoon = []

        print(os.path.realpath(__file__))
        self.__data_path = os.path.dirname(os.path.realpath(__file__))
        self.__number_of_users = 0

        if (ground_truth_dir_path is None):            
            for subdir, dirs, files in os.walk(self.__data_path+"/Data_Mining_Assign1Data/groundTruth"):
                for file in files:                                        
                    self.__ground_truth_file_list.append(os.path.join(subdir, file))
                    self.__number_of_users = self.__number_of_users + 1
            
        # need to allocate those lists
        print("********************************************")
        print("Number of users is  ", self.__number_of_users/2)
        print("Number of eating lists  ", self.__number_of_users)
        print("Total number of lists  ", self.__number_of_users*2)
        print("********************************************")

        self.__myo_data_file_list = []

        if (myo_data_dir_path is None):                        
            for subdir, dirs, files in os.walk(self.__data_path+"/Data_Mining_Assign1Data/MyoData"):
                for file in files:                    
                    self.__myo_data_file_list.append(os.path.join(subdir, file))
        
    def cleanup(self):
        print ("Cleanup")

    def get_number(self):
        self.__figure_number = self.__figure_number + 1
        return self.__figure_number
        
    def preprocessing(self):        
        out_dir = 'out'        
        fork = 'fork'
        spoon = 'spoon'
        eating = 'eating'
        non_eating = 'non_eating'

        try:
            os.mkdir(out_dir)
            os.mkdir(self.__plots_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                print("Actitivty Recognition terminated. Can not proceed since " + out_dir + " and " + self.__plots_dir + " directories exists. Please delete them and run again") 
                sys.exit()
            else:
                raise

        # @todo - delete out directory if exists
        for i in range (len(self.__myo_data_file_list)):
            file = self.__myo_data_file_list[i]
            if "IMU" in file:
                if "fork" in file:
                    # use regex to get the user id of fork IMU data
                    result = re.search('MyoData/(.*)/fork', file)
                    # save the user id
                    self.__imu_myo_fork_data_userid.append(result.group(1))
                    # read the IMU data
                    imu_myo_fork_data=np.genfromtxt(file, dtype=float, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10))
                    # save the IMU data
                    self.__all_imu_myo_fork_data.append(imu_myo_fork_data)
                elif "spoon" in file:
                    # use regex to get the user id of spoon IMU data
                    # repeat the same steps as above for the spoon data
                    result = re.search('MyoData/(.*)/spoon', file)
                    self.__imu_myo_spoon_data_userid.append(result.group(1))
                    imu_myo_spoon_data=np.genfromtxt(file, dtype=float, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10))
                    self.__all_imu_myo_spoon_data.append(imu_myo_spoon_data)
                else:
                    print("unknown file.continue")


        self.create_plot_dir()

        # plot user17 raw data for illustration purposes
        if self.__plot == True:
            user_all_imu_data = self.__all_imu_myo_spoon_data[17]  
            plt.figure(self.get_number())
            plt.plot(user_all_imu_data[:,1], label='x')
            plt.plot(user_all_imu_data[:,2], label='y')
            plt.plot(user_all_imu_data[:,3], label='z')
            plt.plot(user_all_imu_data[:,4], label='w')
            plt.title('IMU Orientation(Queternion) vs sample number')
            plt.ylabel('Orientation')
            plt.xlabel('Sample')
            plt.legend()
            plt.savefig(self.__raw_plots_dir_path + str(self.__figure_number) + '_' + 'IMU Orientation(Queternion) vs sample number'+".png")
            
            plt.figure(self.get_number())
            plt.plot(user_all_imu_data[:,4], label='x')
            plt.plot(user_all_imu_data[:,5], label='y')
            plt.plot(user_all_imu_data[:,6], label='z')
            plt.title('Acceleromter VS sample number')
            plt.ylabel('Acceleromter')
            plt.xlabel('Sample number')
            plt.savefig(self.__raw_plots_dir_path + str(self.__figure_number) + '_' + 'Acceleromter VS sample number'+".png")
            plt.legend()
            
            plt.figure(self.get_number())
            plt.plot(user_all_imu_data[:,7], label='x')
            plt.plot(user_all_imu_data[:,8], label='y')
            plt.plot(user_all_imu_data[:,9], label='z')            
            plt.title('Gyroscope VS sample number')
            plt.ylabel('Gyroscope')
            plt.xlabel('Sample number')
            plt.savefig(self.__raw_plots_dir_path + str(self.__figure_number) + '_' + 'Gyroscope VS sample number'+".png")
            plt.legend()
            
        # Fork
        all_ground_truth_fork_data = []
        ground_truth_fork_data_userid = []
        # Spoon
        all_ground_truth_spoon_data = []
        ground_truth_spoon_data_userid = []

        # Iterate the ground truth data of spoon and fork
        for i in range (len(self.__ground_truth_file_list)):
            file = self.__ground_truth_file_list[i]
            if "fork" in file:
                result = re.search('groundTruth/(.*)/fork', file)
                ground_truth_fork_data_userid.append(result.group(1))
                ground_truth_fork_data=np.genfromtxt(file, dtype=None, delimiter=',', usecols=(0, 1))
                all_ground_truth_fork_data.append(ground_truth_fork_data)
            elif "spoon" in file:
                result = re.search('groundTruth/(.*)/spoon', file)
                ground_truth_spoon_data_userid.append(result.group(1))
                ground_truth_spoon_data=np.genfromtxt(file, dtype=None, delimiter=',', usecols=(0, 1))
                all_ground_truth_spoon_data.append(ground_truth_spoon_data)
            else:
                print("unknown file.continue")

        file_prefix = 'eating_imu_data_'
        non_eating_file_prefix = 'noneating_imu_data_'
        userid = []

        print("\n")
        print("###########################################################################")
        print("Extracting eating with fork IMU data...")
        print("###########################################################################")
        print("\n")

        for i in range (len(all_ground_truth_fork_data)):
            fork_data = all_ground_truth_fork_data[i]
            userid = ground_truth_fork_data_userid[i]
            # Naming incossitency in the data for this project.
            # override in thi case only
            if userid == 'user9':
                userid = 'user09'
            #print(userid)
            #  get the index for this user id
            for k in range (np.size(self.__imu_myo_fork_data_userid)):
                if (userid == self.__imu_myo_fork_data_userid[k]):
                    user_dir_path_fork = out_dir + '/' + userid + '/' + 'fork/'
                    os.makedirs(user_dir_path_fork)
                    uid = k
                    #print("FOUND, ",k)
                    break
            
            myo_fork_data = self.__all_imu_myo_fork_data[k]

            # create a temp numpy array of zeros
            eating_action_range = np.zeros([len(fork_data),2], dtype=int)            

            # should get the size according to the shape
            temp_fork_eating = np.ones((1,10),dtype=float)
            #print(temp_fork_eating)
            temp_fork_non_eating = np.ones((1,10),dtype=float)

            for j in range(len(fork_data)):
                start_frame = fork_data[j][0]
                end_frame = fork_data[j][1]
                
                # the // to competible with python 3
                start_row = (start_frame * 50)//30
                end_row = (end_frame * 50)//30
                eating_action_range[j][0] = start_row
                eating_action_range[j][1] = end_row
                extracted_imu_data = myo_fork_data[start_row:end_row]
                temp_fork_eating = np.concatenate((temp_fork_eating,extracted_imu_data))
            
            # create the range array of non eating fork data
            non_eating_action_row_indexs = np.zeros(len(myo_fork_data),dtype=int)                        

            for j in range(len(eating_action_range)):
                eating_start = eating_action_range[j][0]
                eating_end = eating_action_range[j][1]
                non_eating_action_row_indexs[eating_start:eating_end] = 1

                        
            #  get the indexes of the non eating actions
            extracted_imu_data = []
            non_eating_action_row_indexs = np.where(non_eating_action_row_indexs == 0)[0]
            extracted_imu_data = myo_fork_data[non_eating_action_row_indexs]
            self.__imu_fork_non_eating.append(extracted_imu_data)
            
            if self.__save_files == True:
                # save to file the non eating with fork IMU data
                np.savetxt(user_dir_path_fork + non_eating_file_prefix + '.csv',extracted_imu_data, delimiter=',')

            # make sure to delete the first row of ones
            temp_fork_eating = np.delete(temp_fork_eating,0,0)

            if self.__save_files == True:
                # Save the IMU data for fork eating            
                np.savetxt(user_dir_path_fork + file_prefix + '.csv',temp_fork_eating, delimiter=',')
            
            self.__imu_fork_eating.append(temp_fork_eating)
            temp_fork_eating = []


        
        print("###########################################################################")
        print("Extracting eating with spoon IMU data...")
        print("###########################################################################")

        for i in range (len(all_ground_truth_spoon_data)):
            spoon_data = all_ground_truth_spoon_data[i]
            userid = ground_truth_spoon_data_userid[i]
            #print(userid)
            #  get the index for this user id
            for k in range (np.size(self.__imu_myo_spoon_data_userid)):
                if (userid == self.__imu_myo_spoon_data_userid[k]):
                    user_dir_path_spoon = out_dir + '/' + userid + '/' + 'spoon/'
                    #print user_dir_path_spoon
                    os.makedirs(user_dir_path_spoon)
                    uid = k
                    #print("FOUND, ",k)
                    break
            
            myo_spoon_data = self.__all_imu_myo_spoon_data[k]

            # create a temp numpy array of zeros
            eating_action_range = np.zeros([len(spoon_data),2], dtype=int)
            # should get the size according to the shape
            temp_spoon_eating = np.ones((1,10),dtype=float)
            #print(temp_spoon_eating)   
            temp_spoon_non_eating = np.ones((1,10))

            for j in range(len(spoon_data)):
                start_frame = spoon_data[j][0]
                end_frame = spoon_data[j][1]
                # the // to competible with python 3
                start_row = (start_frame * 50)//30
                end_row = (end_frame * 50)//30
                eating_action_range[j][0] = start_row
                eating_action_range[j][1] = end_row
                extracted_imu_data = myo_spoon_data[start_row:end_row]
                temp_spoon_eating = np.concatenate((temp_spoon_eating,extracted_imu_data))
            

            non_eating_action_row_indexs = []
            
            # create the range array of non eating fork data
            non_eating_action_row_indexs = np.zeros(len(myo_spoon_data),dtype=int)                        

            for j in range(len(eating_action_range)):
                eating_start = eating_action_range[j][0]
                eating_end = eating_action_range[j][1]
                non_eating_action_row_indexs[eating_start:eating_end] = 1

                        
            #  get the indexes of the non eating actions
            extracted_imu_data = []
            non_eating_action_row_indexs = np.where(non_eating_action_row_indexs == 0)[0]
            extracted_imu_data = myo_spoon_data[non_eating_action_row_indexs]
            self.__imu_spoon_non_eating.append(extracted_imu_data)
            
            # save to file the non eating with fork IMU data
            np.savetxt(user_dir_path_spoon + non_eating_file_prefix + '.csv',extracted_imu_data, delimiter=',')


            # make sure to delete the first row of ones
            temp_spoon_eating = np.delete(temp_spoon_eating,0,0)

            # Save the IMU data for fork eating            
            np.savetxt(user_dir_path_spoon + file_prefix + '.csv',temp_spoon_eating, delimiter=',')
            self.__imu_spoon_eating.append(temp_spoon_eating)
            temp_spoon_eating = []
        


    def feature_extraction(self):   
        self.rms()
        self.mean()
        self.variance()
        self.min()
        self.max()
        
        
    def rms(self):
        print("\n")
        print("###########################################################################")
        print("Feature extraction: Calculate RMS of eating and non eating")
        print("###########################################################################")
        

        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            rms = np.sqrt(np.mean(eating ** 2,axis = 0))
            eating_fork[i] = rms

            non_eating = self.__imu_fork_non_eating[i]
            rms = np.sqrt(np.mean(non_eating ** 2,axis = 0))
            non_eating_fork[i] = rms

            eating = self.__imu_spoon_eating[i]
            rms = np.sqrt(np.mean(eating ** 2,axis = 0))
            eating_spoon[i] = rms

            non_eating = self.__imu_spoon_non_eating[i]
            rms = np.sqrt(np.mean(non_eating ** 2,axis = 0))
            non_eating_spoon[i] = rms
            
        self.__rms_eating_fork = eating_fork
        self.__rms_non_eating_fork = non_eating_fork
        self.__rms_eating_spoon = eating_spoon
        self.__rms_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_rms()

    def mean(self):
        print("\n")            
        print("###########################################################################")
        print("Feature extraction: Calculate Mean of eating and non eating...")
        print("###########################################################################")
        
        
        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            mean = np.mean(eating,axis = 0)
            eating_fork[i] = mean

            non_eating = self.__imu_fork_non_eating[i]
            mean = np.mean(non_eating,axis = 0)
            non_eating_fork[i] = mean

            eating = self.__imu_spoon_eating[i]
            mean = np.mean(eating,axis = 0)
            eating_spoon[i] = mean

            non_eating = self.__imu_spoon_non_eating[i]
            mean = np.mean(non_eating,axis = 0)
            non_eating_spoon[i] = mean
            
        self.__mean_eating_fork = eating_fork
        self.__mean_non_eating_fork = non_eating_fork
        self.__mean_eating_spoon = eating_spoon
        self.__mean_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_mean()
                 

    def variance(self):
        print("\n")
        print("###########################################################################")
        print("Feature extraction: Calculate Variance of eating and non eating")
        print("###########################################################################")
        

        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            mean = np.var(eating,axis = 0)
            eating_fork[i] = mean

            non_eating = self.__imu_fork_non_eating[i]
            mean = np.var(non_eating,axis = 0)
            non_eating_fork[i] = mean

            eating = self.__imu_spoon_eating[i]
            mean = np.var(eating,axis = 0)
            eating_spoon[i] = mean

            non_eating = self.__imu_spoon_non_eating[i]
            mean = np.var(non_eating,axis = 0)
            non_eating_spoon[i] = mean
            
        self.__variance_eating_fork = eating_fork
        self.__variance_non_eating_fork = non_eating_fork
        self.__variance_eating_spoon = eating_spoon
        self.__variance_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_variance()

    def min(self):    
        print("\n")
        print("###########################################################################")
        print("Feature extraction: Calculate MIN of eating and non eating...")
        print("###########################################################################")
        

        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        # calculate the MIN acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            min = np.min(eating,axis = 0)
            eating_fork[i] = min

            non_eating = self.__imu_fork_non_eating[i]
            min = np.min(non_eating,axis = 0)
            non_eating_fork[i] = min

            eating = self.__imu_spoon_eating[i]
            min = np.min(eating,axis = 0)
            eating_spoon[i] = min

            non_eating = self.__imu_spoon_non_eating[i]
            min = np.min(non_eating,axis = 0)
            non_eating_spoon[i] = min
            
        self.__min_eating_fork = eating_fork
        self.__min_non_eating_fork = non_eating_fork
        self.__min_eating_spoon = eating_spoon
        self.__min_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_min()

    def max(self):    
        print("\n")
        print("###########################################################################")
        print("Feature extraction: Calculate MAX of eating and non eating...")
        print("###########################################################################")        

        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            max = np.max(eating,axis = 0)
            eating_fork[i] = max

            non_eating = self.__imu_fork_non_eating[i]
            max = np.max(non_eating,axis = 0)
            non_eating_fork[i] = max

            eating = self.__imu_spoon_eating[i]
            max = np.max(eating,axis = 0)
            eating_spoon[i] = max

            non_eating = self.__imu_spoon_non_eating[i]
            max = np.min(non_eating,axis = 0)
            non_eating_spoon[i] = max
            
        self.__max_eating_fork = eating_fork
        self.__max_non_eating_fork = non_eating_fork
        self.__max_eating_spoon = eating_spoon
        self.__max_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_max()


    
    def create_plot_dir(self):
        plots_dir_path = self.__plots_dir + '/' + 'RMS' + '/'
        os.makedirs(plots_dir_path)
        plots_dir_path = self.__plots_dir + '/' + 'Mean' + '/'
        os.makedirs(plots_dir_path)
        plots_dir_path = self.__plots_dir + '/' + 'Variance' + '/'
        os.makedirs(plots_dir_path)
        plots_dir_path = self.__plots_dir + '/' + 'MIN' + '/'
        os.makedirs(plots_dir_path)
        plots_dir_path = self.__plots_dir + '/' + 'MAX' + '/'
        os.makedirs(plots_dir_path)
        
        self.__raw_plots_dir_path = self.__plots_dir + '/' + 'user_raw_imu_data' + '/'
        os.makedirs(self.__raw_plots_dir_path)

        self.__pca_plots_dir_path = self.__plots_dir + '/' + 'PCA' + '/'
        os.makedirs(self.__pca_plots_dir_path)


    def plot_rms(self):        
        feture='RMS'
        self.plot_orientation(feture,self.__rms_eating_fork,self.__rms_non_eating_fork,self.__rms_eating_spoon,self.__rms_non_eating_spoon)
        self.plot_acceleration(feture,self.__rms_eating_fork,self.__rms_non_eating_fork,self.__rms_eating_spoon,self.__rms_non_eating_spoon)        
        self.plot_gyro(feture,self.__rms_eating_fork,self.__rms_non_eating_fork,self.__rms_eating_spoon,self.__rms_non_eating_spoon)


    def plot_mean(self):
        feture='Mean'
        self.plot_orientation(feture,self.__mean_eating_fork,self.__mean_non_eating_fork,self.__mean_eating_spoon,self.__mean_non_eating_spoon)
        self.plot_acceleration(feture,self.__mean_eating_fork,self.__mean_non_eating_fork,self.__mean_eating_spoon,self.__mean_non_eating_spoon)
        self.plot_gyro(feture,self.__mean_eating_fork,self.__mean_non_eating_fork,self.__mean_eating_spoon,self.__mean_non_eating_spoon)


    def plot_variance(self):
        feture='Variance'
        self.plot_orientation(feture,self.__variance_eating_fork,self.__variance_non_eating_fork,self.__variance_eating_spoon,self.__variance_non_eating_spoon)
        self.plot_acceleration(feture,self.__variance_eating_fork,self.__variance_non_eating_fork,self.__variance_eating_spoon,self.__variance_non_eating_spoon)
        self.plot_gyro(feture,self.__variance_eating_fork,self.__variance_non_eating_fork,self.__variance_eating_spoon,self.__variance_non_eating_spoon)

    def plot_min(self):    
        feture='MIN'
        self.plot_orientation(feture,self.__min_eating_fork,self.__min_non_eating_fork,self.__min_eating_spoon,self.__min_non_eating_spoon)
        self.plot_acceleration(feture,self.__min_eating_fork,self.__min_non_eating_fork,self.__min_eating_spoon,self.__min_non_eating_spoon)
        self.plot_gyro(feture,self.__min_eating_fork,self.__min_non_eating_fork,self.__min_eating_spoon,self.__min_non_eating_spoon)

    def plot_max(self):    
        feture='MAX'
        self.plot_orientation(feture,self.__max_eating_fork,self.__max_non_eating_fork,self.__max_eating_spoon,self.__max_non_eating_spoon)
        self.plot_acceleration(feture,self.__max_eating_fork,self.__max_non_eating_fork,self.__max_eating_spoon,self.__max_non_eating_spoon)
        self.plot_gyro(feture,self.__max_eating_fork,self.__max_non_eating_fork,self.__max_eating_spoon,self.__max_non_eating_spoon)

    def plot_orientation(self,feature,eating_fork,non_eating_fork,eating_spoon, non_eating_spoon):    
        plots_dir_path = self.__plots_dir + '/' + feature + '/' 
        plt.figure(self.get_number())
        plt.plot(eating_fork[:,0], label='x')
        plt.plot(eating_fork[:,1], label='y')
        plt.plot(eating_fork[:,2], label='z')
        plt.plot(eating_fork[:,3], label='w')
        plt.title('Orientation ' + feature + '- Eating(fork)')
        plt.ylabel('Orientation ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Orientation ' + feature + '- Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(non_eating_fork[:,0], label='x')
        plt.plot(non_eating_fork[:,1], label='y')
        plt.plot(non_eating_fork[:,2], label='z')
        plt.plot(non_eating_fork[:,3], label='w')
        plt.title('Orientation ' + feature + '- Non Eating(fork)')
        plt.ylabel('Orientation ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Orientation ' + feature + '- Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(eating_spoon[:,0], label='x')
        plt.plot(eating_spoon[:,1], label='y')
        plt.plot(eating_spoon[:,2], label='z')
        plt.plot(eating_spoon[:,3], label='w')
        plt.title('Orientation ' + feature + '- Eating(spoon)')
        plt.ylabel('Orientation ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Orientation ' + feature + '- Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(non_eating_spoon[:,0], label='x')
        plt.plot(non_eating_spoon[:,1], label='y')
        plt.plot(non_eating_spoon[:,2], label='z')
        plt.plot(non_eating_spoon[:,3], label='w')
        plt.title('Orientation ' + feature + '- Non Eating(spoon)')
        plt.ylabel('Orientation ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Orientation ' + feature + '- Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        if self.__show_plot == True:
            plt.show() 

    def plot_acceleration(self,feature,eating_fork,non_eating_fork,eating_spoon, non_eating_spoon):        
        plots_dir_path = self.__plots_dir + '/' + feature + '/' 
        plt.figure(self.get_number())
        plt.plot(eating_fork[:,4], label='x')
        plt.plot(eating_fork[:,5], label='y')
        plt.plot(eating_fork[:,6], label='z')
        plt.title('Acceleration ' + feature + '- Eating(fork)')
        plt.ylabel('Acceleration ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Acceleration ' + feature + '- Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(non_eating_fork[:,4], label='x')
        plt.plot(non_eating_fork[:,5], label='y')
        plt.plot(non_eating_fork[:,6], label='z')
        plt.title('Acceleration ' + feature + '- Non Eating(fork)')
        plt.ylabel('Acceleration ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Acceleration ' + feature + '- Non Eating(fork)'+".png")
        plt.close(self.__figure_number)


        plt.figure(self.get_number())
        plt.plot(eating_spoon[:,4], label='x')
        plt.plot(eating_spoon[:,5], label='y')
        plt.plot(eating_spoon[:,6], label='z')
        plt.title('Acceleration ' + feature + '- Eating(spoon)')
        plt.ylabel('Acceleration ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Acceleration ' + feature + '- Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(non_eating_spoon[:,4], label='x')
        plt.plot(non_eating_spoon[:,5], label='y')
        plt.plot(non_eating_spoon[:,6], label='z')
        plt.title('Acceleration ' + feature + '- Non Eating(spoon)')
        plt.ylabel('Acceleration ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Acceleration ' + feature + '- Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        if self.__show_plot == True:
            plt.show()             

    def plot_gyro(self,feature,eating_fork,non_eating_fork,eating_spoon, non_eating_spoon):
        plots_dir_path = self.__plots_dir + '/' + feature + '/'         
        plt.figure(self.get_number())
        plt.plot(eating_fork[:,7], label='x')
        plt.plot(eating_fork[:,8], label='y')
        plt.plot(eating_fork[:,9], label='z')
        plt.title('Gyroscope ' + feature + '- Eating(fork)')
        plt.ylabel('Gyroscope ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Gyroscope ' + feature + '- Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(non_eating_fork[:,7], label='x')
        plt.plot(non_eating_fork[:,8], label='y')
        plt.plot(non_eating_fork[:,9], label='z')
        plt.title('Gyroscope ' + feature + '- Non Eating(fork)')
        plt.ylabel('Gyroscope ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Gyroscope ' + feature + '- Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(eating_spoon[:,7], label='x')
        plt.plot(eating_spoon[:,8], label='y')
        plt.plot(eating_spoon[:,9], label='z')
        plt.title('Gyroscope ' + feature + '- Eating(spoon)')
        plt.ylabel('Gyroscope ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Gyroscope ' + feature + '- Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(non_eating_spoon[:,7], label='x')
        plt.plot(non_eating_spoon[:,8], label='y')
        plt.plot(non_eating_spoon[:,9], label='z')
        plt.title('Gyroscope ' + feature + '- Non Eating(spoon)')
        plt.ylabel('Gyroscope ' + feature)
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(plots_dir_path + str(self.__figure_number) + '_' + 'Gyroscope ' + feature + '- Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 


    def pca(self):        
        print("\n")
        print("###########################################################################")
        print("Feature Selection: PCA")
        print("###########################################################################")
        
        # Try this tommorow
        # https://stackabuse.com/dimensionality-reduction-in-python-with-scikit-learn/
        

        ## https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
        ## https://plot.ly/python/v3/ipython-notebooks/principal-component-analysis/
        ##https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/
        # fork eating and non eating

        #30x50
        X_1 = np.hstack\
            ((self.__mean_eating_fork,\
            self.__variance_eating_fork, \
            self.__rms_eating_fork, \
            self.__min_eating_fork, \
            self.__max_eating_fork  ))
        #30x50
        X_2 = np.hstack\
            ((self.__mean_non_eating_fork, \
            self.__variance_non_eating_fork, \
            self.__rms_non_eating_fork, \
            self.__min_non_eating_fork, \
            self.__max_non_eating_fork))

        #60x50
        X_fork = np.vstack((X_1, X_2))


        # spoon eating and non eating
        X_spoon = np.hstack\
            ((self.__mean_eating_spoon,\
            self.__variance_eating_spoon, \
            self.__rms_eating_spoon, \
            self.__min_eating_spoon, \
            self.__max_eating_spoon, \
            self.__mean_non_eating_spoon, \
            self.__variance_non_eating_spoon, \
            self.__rms_non_eating_spoon, \
            self.__min_non_eating_spoon, \
            self.__max_non_eating_spoon))

        # Let's scale the feature set before doing pca
        # here is an example what would happened without scaling
        #https://scikit-learn.org/stable/modules/preprocessing.html
        X_fork_scaled = StandardScaler().fit_transform(X_fork)
        X_spoon_scaled = StandardScaler().fit_transform(X_spoon)

        number_of_components = 10
        pca_X_fork = PCA(number_of_components)

        pca_trans = pca_X_fork.fit_transform(X_fork)

        # Scree Plot
        per_var = np.round(pca_X_fork.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range (1,len(per_var)+1)]
        x = np.arange(1,len(per_var)+1)
        plt.figure(self.get_number())
        plt.bar(x,per_var)
        plt.xticks(x,labels)
        plt.ylabel('Percentage of Explained Variance of X fork')
        plt.xlabel('principal component')
        plt.title('Scree plot')
        plt.show()
        plt.savefig(self.__pca_plots_dir_path + str(self.__figure_number) + '_' + 'PCA Scree Plot'+".png")
        
        colors = {'eating': '#0D76BF', 'non-eating': '#00cc96'}

        y = np.zeros([60,1])
        data = []

        for name, col in zip(('eating', 'non-eating'), colors.values()):

            trace = dict(
                type='scatter',
                x=pca_trans[y==name,0],
                y=pca_trans[y==name,1],
                mode='markers',
                name=name,
                marker=dict(
                    color=col,
                    size=12,
                    line=dict(
                        color='rgba(217, 217, 217, 0.14)',
                        width=0.5),
                    opacity=0.8)
            )
            data.append(trace)

        layout = dict(
                xaxis=dict(title='PC1', showline=False),
                yaxis=dict(title='PC2', showline=False)
        )
        fig = dict(data=data, layout=layout)
        py.iplot(fig, filename='pca-scikitlearn')
                #plt.figure(self.get_number())
        #plt.plot(pca_trans[0:50,0],pca_trans[0:50,1], 'o', markersize=7, color='blue', alpha=0.5, label='eating')
        #plt.plot(pca_trans[50:100,0], pca_trans[50:100,1], '^', markersize=7, color='red', alpha=0.5, label='non_eating')
        #plt.xlabel('x_values')
        #plt.ylabel('y_values')
        #plt.legend()
        #plt.title('Transformed samples with class labels')
        #plt.show()
        #plt.savefig(self.__pca_plots_dir_path + str(self.__figure_number) + '_' + 'PCA Transformed with class label'+".png")

def main():        
    activityRecognition = ActivityRecognition()
    activityRecognition.preprocessing()
    activityRecognition.feature_extraction()
    activityRecognition.pca()

if __name__ == "__main__":
    main()
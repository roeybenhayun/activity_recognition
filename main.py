import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from tempfile import TemporaryFile
import re



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

        ## Mean
        # todo - replace the above with : self.__accelerometer_mean
        self.__acceleration_mean_eating_fork = []
        self.__acceleration_mean_non_eating_fork = []
        self.__acceleration_mean_eating_spoon = []
        self.__acceleration_mean_non_eating_spoon = []

        # todo - replace the above with : self.__gyro_mean
        self.__gyro_mean_eating_fork = []
        self.__gyro_mean_non_eating_fork = []
        self.__gyro_mean_eating_spoon = []
        self.__gyro_mean_non_eating_spoon = []

        # todo - replace the above with : self.__orientation_mean
        self.__orientation_mean_eating_fork = []
        self.__orientation_mean_non_eating_fork = []
        self.__orientation_mean_eating_spoon = []
        self.__orientation_mean_non_eating_spoon = []

        ## Acceleration
        # todo - replace the above with : self.__acceleration_variance
        self.__acceleration_variance_eating_fork = []
        self.__acceleration_variance_non_eating_fork = []
        self.__acceleration_variance_eating_spoon= []
        self.__acceleration_variance_non_eating_spoon = []

        # todo - replace the above with : self.__gyro_variance
        self.__gyro_variance_eating_fork = []
        self.__gyro_variance_non_eating_fork = []
        self.__gyro_variance_eating_spoon = []
        self.__gyro_variance_non_eating_spoon = []

        # todo - replace the above with : self.__orientation_variance
        self.__orientation_variance_eating_fork = []
        self.__orientation_variance_non_eating_fork = []
        self.__orientation_variance_eating_spoon = []
        self.__orientation_variance_non_eating_spoon = []

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
        
        #FFT
        self.__fft_eating_fork = []
        self.__fft_non_eating_fork = []
        self.__fft_eating_spoon= []
        self.__fft_non_eating_spoon = []

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
        
        # output directory for results
        # should be an option?
        os.mkdir(out_dir)

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

        # @todo add print flag here
        # by default this flag should be disabled
        # only print relevant data
        print("IMU MYO SPOON DATA USERS")
        print(self.__imu_myo_spoon_data_userid)
        print("IMU MYO FORK DATA USERS")
        print(self.__imu_myo_spoon_data_userid)

        
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
            plt.savefig(str(self.__figure_number) + '_' + 'IMU Orientation(Queternion) vs sample number'+".png")
            
            plt.figure(self.get_number())
            plt.plot(user_all_imu_data[:,4], label='x')
            plt.plot(user_all_imu_data[:,5], label='y')
            plt.plot(user_all_imu_data[:,6], label='z')
            plt.title('Acceleromter VS sample number')
            plt.ylabel('Acceleromter')
            plt.xlabel('Sample number')
            plt.savefig(str(self.__figure_number) + '_' + 'Acceleromter VS sample number'+".png")
            plt.legend()
            
            plt.figure(self.get_number())
            plt.plot(user_all_imu_data[:,7], label='x')
            plt.plot(user_all_imu_data[:,8], label='y')
            plt.plot(user_all_imu_data[:,9], label='z')            
            plt.title('Gyroscope VS sample number')
            plt.ylabel('Gyroscope')
            plt.xlabel('Sample number')
            plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope VS sample number'+".png")
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


        print("GROUND TRUTH SPOON USERS")
        print(ground_truth_spoon_data_userid)
        print("GROUND TRUTH FORK USERS")
        print(ground_truth_fork_data_userid)

        file_prefix = 'eating_imu_data_'
        non_eating_file_prefix = 'noneating_imu_data_'
        userid = []
        ###########################################################################
        # Fork IMU data
        ###########################################################################
        print("###########################################################################")
        print("Extracting eating with fork IMU data...")
        print("###########################################################################")
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
                #print ("Start = ", start_frame)
                #print ("End = ", end_frame)
                start_row = (start_frame * 50)/30
                end_row = (end_frame * 50)/30
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

        # @todo: merge to one function
        ###########################################################################
        # Spoon IMU data
        ###########################################################################
        
        print("###########################################################################")
        print("Extracting eating with spoon IMU data...")
        print("###########################################################################")
        #file_prefix = 'eating_with_spoon_imu_data_'
        #non_eating_file_prefix = 'noneating_with_spoon_imu_data_'

        for i in range (len(all_ground_truth_spoon_data)):
            spoon_data = all_ground_truth_spoon_data[i]
            userid = ground_truth_spoon_data_userid[i]
            #print(userid)
            #  get the index for this user id
            for k in range (np.size(self.__imu_myo_spoon_data_userid)):
                if (userid == self.__imu_myo_spoon_data_userid[k]):
                    user_dir_path_spoon = out_dir + '/' + userid + '/' + 'spoon/'
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
                start_row = (start_frame * 50)/30
                end_row = (end_frame * 50)/30
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
        
        print("###########################################################################")
        print("Extracting eating actions IMU data completed")
        print("###########################################################################")


    def feature_extraction(self):
        """
        Feature extraction step
        Parameters
        ----------
        None

        Return
        ------
        None
        """        
        self.rms()
        self.mean()
        self.variance()
        #self.fft()
        #self.fft()
        #self.min()
        #self.max()
        
        #self.energy_spectral_density()
        #self.acceleration_mean()
        #self.acceleration_variance()
        #self.gyro_mean()        
        #self.gyro_variance()
        #self.orientation_mean()
        #self.orientation_variance()


    def rms(self):

        print("\n###########################################################################")
        print("Calculate RMS of eating and non eating")
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

        print("\n###########################################################################")
        print("Calculate Mean of eating and non eating")
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

        print("\n###########################################################################")
        print("Calculate Mean of eating and non eating")
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
          
    def fft(self):

        print("\n###########################################################################")
        print("Calculate FFT of eating and non eating")
        print("###########################################################################")
        
        eating_fork = np.zeros([len(self.__imu_fork_eating),10],dtype=float)
        non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),10],dtype=float) 
        eating_spoon = np.zeros([len(self.__imu_spoon_eating),10],dtype=float)
        non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),10],dtype=float)
        
        t1 = self.__imu_fork_eating

        for i in range(len(self.__imu_fork_eating)):
            eating = self.__imu_fork_eating[i]
            fft = np.fft.fft(eating,axis = 0)
            t1[i] = fft

            #non_eating = self.__imu_fork_non_eating[i]
            #fft = np.fft.fft(non_eating,axis = 0)
            #non_eating_fork[i] = fft
#
            #eating = self.__imu_spoon_eating[i]
            #fft = np.fft.fft(eating,axis = 0)
            #eating_spoon[i] = fft
#
            #non_eating = self.__imu_spoon_non_eating[i]
            #fft = np.fft.fft(non_eating,axis = 0)
            #non_eating_spoon[i] = fft
            
        self.__fft_eating_fork = eating_fork
        self.__fft_non_eating_fork = non_eating_fork
        self.__fft_eating_spoon = eating_spoon
        self.__fft_non_eating_spoon = non_eating_spoon

        if self.__plot == True:
            self.plot_fft()


    def orientation_mean(self):
        """
        Calculate the mean Gyro for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        mean gyro 
        """
        print("\n###########################################################################")
        print("Calculate Gyro mean of eating and non eating")
        print("###########################################################################")
        
        orientation_mean_eating_fork = np.zeros([len(self.__imu_fork_eating),4],dtype=float)
        orientation_mean_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),4],dtype=float) 
        orientation_mean_eating_spoon = np.zeros([len(self.__imu_spoon_eating),4],dtype=float)
        orientation_mean_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),4],dtype=float)
        
        x = self.__orientation_axis.get("x")
        y = self.__orientation_axis.get("y")
        z = self.__orientation_axis.get("z")
        w = self.__orientation_axis.get("w")
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            orientation_mean = np.mean(user_fork_eating,axis = 0)
            orientation_mean_eating_fork[i][0] = orientation_mean[x]
            orientation_mean_eating_fork[i][1] = orientation_mean[y]
            orientation_mean_eating_fork[i][2] = orientation_mean[z]
            orientation_mean_eating_fork[i][3] = orientation_mean[w]

            user_fork_non_eating = self.__imu_fork_non_eating[i]
            orientation_mean = np.mean(user_fork_non_eating,axis = 0)          
            orientation_mean_non_eating_fork[i][0] = orientation_mean[x]
            orientation_mean_non_eating_fork[i][1] = orientation_mean[y]
            orientation_mean_non_eating_fork[i][2] = orientation_mean[z]
            orientation_mean_non_eating_fork[i][3] = orientation_mean[w]

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]
            orientation_mean = np.mean(user_eating,axis = 0)
            orientation_mean_eating_spoon[i][0] = orientation_mean[x]
            orientation_mean_eating_spoon[i][1] = orientation_mean[y]
            orientation_mean_eating_spoon[i][2] = orientation_mean[z]
            orientation_mean_eating_spoon[i][3] = orientation_mean[w]

            user_non_eating = self.__imu_spoon_non_eating[i]
            orientation_mean = np.mean(user_non_eating,axis = 0)          
            orientation_mean_non_eating_spoon[i][0] = orientation_mean[x]
            orientation_mean_non_eating_spoon[i][1] = orientation_mean[y]
            orientation_mean_non_eating_spoon[i][2] = orientation_mean[z]
            orientation_mean_non_eating_spoon[i][3] = orientation_mean[w]
            
        self.__orientation_mean_eating_fork = orientation_mean_eating_fork
        self.__orientation_mean_non_eating_fork = orientation_mean_non_eating_fork
        self.__orientation_mean_eating_spoon = orientation_mean_eating_spoon
        self.__orientation_mean_non_eating_spoon = orientation_mean_non_eating_spoon


        if self.__plot == True:
            self.plot_orientation_mean()
        

    def orientation_variance(self):
        """
        Calculate the mean Orientation Variance for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        mean gyro 
        """
        print("\n###########################################################################")
        print("Calculate Orientation variance of eating and non eating activities ")
        print("###########################################################################")
        
        orientation_variance_eating_fork = np.zeros([len(self.__imu_fork_eating),4],dtype=float)
        orientation_variance_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),4],dtype=float) 
        orientation_variance_eating_spoon = np.zeros([len(self.__imu_spoon_eating),4],dtype=float)
        orientation_variance_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),4],dtype=float)
        
        x = self.__orientation_axis.get("x")
        y = self.__orientation_axis.get("y")
        z = self.__orientation_axis.get("z")
        w = self.__orientation_axis.get("w")

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            orientation_variance = np.mean(user_fork_eating,axis = 0)
            orientation_variance_eating_fork[i][0] = orientation_variance[x]
            orientation_variance_eating_fork[i][1] = orientation_variance[y]
            orientation_variance_eating_fork[i][2] = orientation_variance[z]
            orientation_variance_eating_fork[i][3] = orientation_variance[w]

            user_fork_non_eating = self.__imu_fork_non_eating[i]
            orientation_variance = np.mean(user_fork_non_eating,axis = 0)          
            orientation_variance_non_eating_fork[i][0] = orientation_variance[x]
            orientation_variance_non_eating_fork[i][1] = orientation_variance[y]
            orientation_variance_non_eating_fork[i][2] = orientation_variance[z]
            orientation_variance_non_eating_fork[i][3] = orientation_variance[w]

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]
            orientation_variance = np.mean(user_eating,axis = 0)
            orientation_variance_eating_spoon[i][0] = orientation_variance[x]
            orientation_variance_eating_spoon[i][1] = orientation_variance[y]
            orientation_variance_eating_spoon[i][2] = orientation_variance[z]
            orientation_variance_eating_spoon[i][3] = orientation_variance[w]

            user_non_eating = self.__imu_spoon_non_eating[i]
            orientation_variance = np.mean(user_non_eating,axis = 0)          
            orientation_variance_non_eating_spoon[i][0] = orientation_variance[x]
            orientation_variance_non_eating_spoon[i][1] = orientation_variance[y]
            orientation_variance_non_eating_spoon[i][2] = orientation_variance[z]
            orientation_variance_non_eating_spoon[i][3] = orientation_variance[w]
            
        self.__orientation_variance_eating_fork = orientation_variance_eating_fork
        self.__orientation_variance_non_eating_fork = orientation_variance_non_eating_fork
        self.__orientation_variance_eating_spoon = orientation_variance_eating_spoon
        self.__orientation_variance_non_eating_spoon = orientation_variance_non_eating_spoon

        if self.__plot == True:
            self.plot_orientation_variance()

    def plot_orientation_mean(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_fork[:,0], label='x')
        plt.plot(self.__mean_eating_fork[:,1], label='y')
        plt.plot(self.__mean_eating_fork[:,2], label='z')
        plt.plot(self.__mean_eating_fork[:,3], label='w')
        plt.title('Orientation Mean - Eating(fork)')
        plt.ylabel('Orientation Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Orientation Mean - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_fork[:,0], label='x')
        plt.plot(self.__mean_non_eating_fork[:,1], label='y')
        plt.plot(self.__mean_non_eating_fork[:,2], label='z')
        plt.plot(self.__mean_non_eating_fork[:,3], label='w')
        plt.title('Orientation Mean - Non Eating(fork)')
        plt.ylabel('Orientation Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Mean - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_spoon[:,0], label='x')
        plt.plot(self.__mean_eating_spoon[:,1], label='y')
        plt.plot(self.__mean_eating_spoon[:,2], label='z')
        plt.plot(self.__mean_eating_spoon[:,3], label='w')
        plt.title('Orientation Mean - Eating(spoon)')
        plt.ylabel('Orientation Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Mean - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_spoon[:,0], label='x')
        plt.plot(self.__mean_non_eating_spoon[:,1], label='y')
        plt.plot(self.__mean_non_eating_spoon[:,2], label='z')
        plt.plot(self.__mean_non_eating_spoon[:,3], label='w')
        plt.title('Orientation Mean - Non Eating(spoon)')
        plt.ylabel('Orientation Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Orientation Mean - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        if self.__show_plot == True:
            plt.show()   


    def plot_orientation_variance(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_eating_fork[:,0], label='x')
        plt.plot(self.__variance_eating_fork[:,1], label='y')
        plt.plot(self.__variance_eating_fork[:,2], label='z')
        plt.plot(self.__variance_eating_fork[:,3], label='w')
        plt.title('Orientation Variance - Eating(fork)')
        plt.ylabel('Orientation Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Variance - Eating(fork)'+".png")
        plt.close(self.__figure_number)

        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_fork[:,0], label='x')
        plt.plot(self.__variance_non_eating_fork[:,1], label='y')
        plt.plot(self.__variance_non_eating_fork[:,2], label='z')
        plt.plot(self.__variance_non_eating_fork[:,3], label='w')
        plt.title('Orientation Variance - Non Eating(fork)')
        plt.ylabel('Orientation Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Variance - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__variance_eating_spoon[:,0], label='x')
        plt.plot(self.__variance_eating_spoon[:,1], label='y')
        plt.plot(self.__variance_eating_spoon[:,2], label='z')
        plt.plot(self.__variance_eating_spoon[:,3], label='w')
        plt.title('Orientation Variance - Eating(spoon)')
        plt.ylabel('Orientation Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Variance - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_spoon[:,0], label='x')
        plt.plot(self.__variance_non_eating_spoon[:,1], label='y')
        plt.plot(self.__variance_non_eating_spoon[:,2], label='z')
        plt.plot(self.__variance_non_eating_spoon[:,3], label='w')
        plt.title('Orientation Variance - Non Eating(spoon)')
        plt.ylabel('Orientation Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation Variance - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        if self.__show_plot == True:
            plt.show()       


    def plot_rms__(self):
        fetaure = 'RMS'
        sensor = ["Orientation","Acceleration", "Gyro"]
        activity = ["Eating(fork)","Non Eating(fork)","Eating(spoon)","Non Eating(spoon)"]
        print ("foo")
        #for i in len(sensor):
        #    sensor_type = sensor[i]
        #    for j in len(activity):
        #        activity_type  = activity[i]


    def plot_orientation_rms(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__rms_eating_fork[:,0], label='x')
        plt.plot(self.__rms_eating_fork[:,1], label='y')
        plt.plot(self.__rms_eating_fork[:,2], label='z')
        plt.plot(self.__rms_eating_fork[:,3], label='w')
        plt.title('Orientation RMS - Eating(fork)')
        plt.ylabel('Orientation RMS')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation RMS - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_fork[:,0], label='x')
        plt.plot(self.__rms_non_eating_fork[:,1], label='y')
        plt.plot(self.__rms_non_eating_fork[:,2], label='z')
        plt.plot(self.__rms_non_eating_fork[:,3], label='w')
        plt.title('Orientation RMS - Non Eating(fork)')
        plt.ylabel('Orientation RMS')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation RMS - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__rms_eating_spoon[:,0], label='x')
        plt.plot(self.__rms_eating_spoon[:,1], label='y')
        plt.plot(self.__rms_eating_spoon[:,2], label='z')
        plt.plot(self.__rms_eating_spoon[:,3], label='w')
        plt.title('Orientation RMS - Eating(spoon)')
        plt.ylabel('Orientation RMS')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation RMS - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_spoon[:,0], label='x')
        plt.plot(self.__rms_non_eating_spoon[:,1], label='y')
        plt.plot(self.__rms_non_eating_spoon[:,2], label='z')
        plt.plot(self.__rms_non_eating_spoon[:,3], label='w')
        plt.title('Orientation RMS - Non Eating(spoon)')
        plt.ylabel('Orientation RMS')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation RMS - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        if self.__show_plot == True:
            plt.show() 

    def gyro_mean(self):
        """
        Calculate the mean Gyro for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        mean gyro 
        """
        print("\n###########################################################################")
        print("Calculate Gyro mean of eating and non eating")
        print("###########################################################################")
        
        gyro_mean_eating_fork = np.zeros([len(self.__imu_fork_eating),3],dtype=float)
        gyro_mean_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),3],dtype=float) 
        gyro_mean_eating_spoon = np.zeros([len(self.__imu_spoon_eating),3],dtype=float)
        gyro_mean_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),3],dtype=float)
        
        x = self.__gyro_axis.get("x")
        y = self.__gyro_axis.get("y")
        z = self.__gyro_axis.get("z")
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            gyro_mean = np.mean(user_fork_eating,axis = 0)
            gyro_mean_eating_fork[i][0] = gyro_mean[x]
            gyro_mean_eating_fork[i][1] = gyro_mean[y]
            gyro_mean_eating_fork[i][2] = gyro_mean[z]

            user_fork_non_eating = self.__imu_fork_non_eating[i]
            gyro_mean = np.mean(user_fork_non_eating,axis = 0)          
            gyro_mean_non_eating_fork[i][0] = gyro_mean[x]
            gyro_mean_non_eating_fork[i][1] = gyro_mean[y]
            gyro_mean_non_eating_fork[i][2] = gyro_mean[z]


        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]
            # calculate the mean acceleration on the x,y,z axis (columns)
            gyro_mean = np.mean(user_eating,axis = 0)
            gyro_mean_eating_spoon[i][0] = gyro_mean[x]
            gyro_mean_eating_spoon[i][1] = gyro_mean[y]
            gyro_mean_eating_spoon[i][2] = gyro_mean[z]

            user_non_eating = self.__imu_spoon_non_eating[i]
            # calculate the mean acceleration on the x,y,z axis (columns)
            gyro_mean = np.mean(user_non_eating,axis = 0)          
            gyro_mean_non_eating_spoon[i][0] = gyro_mean[x]
            gyro_mean_non_eating_spoon[i][1] = gyro_mean[y]
            gyro_mean_non_eating_spoon[i][2] = gyro_mean[z]
            
        self.__gyro_mean_eating_fork = gyro_mean_eating_fork
        self.__gyro_mean_non_eating_fork = gyro_mean_non_eating_fork
        self.__gyro_mean_eating_spoon = gyro_mean_eating_spoon
        self.__gyro_mean_non_eating_spoon = gyro_mean_non_eating_spoon


        if self.__plot == True:
            self.plot_gyro_mean()

          


    def gyro_variance(self):
        """
        Calculate the mean Gyro for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        variance gyro 
        """
        print("\n###########################################################################")
        print("Calculate Gyro variance")
        print("###########################################################################")
        gyro_variance_fork_eating = np.zeros([len(self.__imu_fork_eating),3],dtype=float)
        gyro_variance_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),3],dtype=float) 
        gyro_variance_spoon_eating = np.zeros([len(self.__imu_spoon_eating),3],dtype=float)
        gyro_variance_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),3],dtype=float)

        x = self.__gyro_axis.get("x")
        y = self.__gyro_axis.get("y")
        z = self.__gyro_axis.get("z")

        # calculate the mean gyro on the x,y,z axis (columns)        
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            gyro_variance = np.var(user_fork_eating,axis = 0)
            #if i == 0:
            # how to slice. 
            # when refactoring - calc the mean and var and just slice each sensor type
            #    print gyro_variance[x:z+1]
            gyro_variance_fork_eating[i][0] = gyro_variance[x]
            gyro_variance_fork_eating[i][1] = gyro_variance[y]
            gyro_variance_fork_eating[i][2] = gyro_variance[z]
            
            user_fork_non_eating = self.__imu_fork_non_eating[i]            
            gyro_variance = np.var(user_fork_non_eating,axis = 0)          
            gyro_variance_non_eating_fork[i][0] = gyro_variance[x]
            gyro_variance_non_eating_fork[i][1] = gyro_variance[y]
            gyro_variance_non_eating_fork[i][2] = gyro_variance[z]

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]            
            gyro_variance = np.var(user_eating,axis = 0)
            gyro_variance_spoon_eating[i][0] = gyro_variance[x]
            gyro_variance_spoon_eating[i][1] = gyro_variance[y]
            gyro_variance_spoon_eating[i][2] = gyro_variance[z]

            user_non_eating = self.__imu_spoon_non_eating[i]            
            gyro_variance = np.var(user_non_eating,axis = 0)          
            gyro_variance_non_eating_spoon[i][0] = gyro_variance[x]
            gyro_variance_non_eating_spoon[i][1] = gyro_variance[y]
            gyro_variance_non_eating_spoon[i][2] = gyro_variance[z]

        self.__gyro_variance_eating_fork = gyro_variance_fork_eating
        self.__gyro_variance_non_eating_fork = gyro_variance_non_eating_fork
        self.__gyro_variance_eating_spoon = gyro_variance_spoon_eating
        self.__gyro_variance_non_eating_spoon = gyro_variance_non_eating_spoon
        
        if self.__plot == True:
            self.plot_gyro_variance() 


    def plot_fft(self):
        self.plot_orientation_fft()
        self.plot_acceleration_fft()        
        self.plot_gyro_fft()

    def plot_rms(self):
        self.plot_orientation_rms()
        self.plot_acceleration_rms()        
        self.plot_gyro_rms()

    def plot_mean(self):
        self.plot_orientation_mean()
        self.plot_acceleration_mean()        
        self.plot_gyro_mean()

    def plot_variance(self):
        self.plot_orientation_variance()
        self.plot_acceleration_variance()
        self.plot_gyro_variance()
        
#################

    def plot_orientation_fft(self):    
        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_fork[:,0], label='x')
        plt.plot(self.__fft_eating_fork[:,1], label='y')
        plt.plot(self.__fft_eating_fork[:,2], label='z')
        plt.plot(self.__fft_eating_fork[:,3], label='w')
        plt.title('Orientation FFT - Eating(fork)')
        plt.ylabel('Orientation FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation FFT - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_fork[:,0], label='x')
        plt.plot(self.__fft_non_eating_fork[:,1], label='y')
        plt.plot(self.__fft_non_eating_fork[:,2], label='z')
        plt.plot(self.__fft_non_eating_fork[:,3], label='w')
        plt.title('Orientation FFT - Non Eating(fork)')
        plt.ylabel('Orientation FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation FFT - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_spoon[:,0], label='x')
        plt.plot(self.__fft_eating_spoon[:,1], label='y')
        plt.plot(self.__fft_eating_spoon[:,2], label='z')
        plt.plot(self.__fft_eating_spoon[:,3], label='w')
        plt.title('Orientation FFT - Eating(spoon)')
        plt.ylabel('Orientation FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation FFT - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_spoon[:,0], label='x')
        plt.plot(self.__fft_non_eating_spoon[:,1], label='y')
        plt.plot(self.__fft_non_eating_spoon[:,2], label='z')
        plt.plot(self.__fft_non_eating_spoon[:,3], label='w')
        plt.title('Orientation FFT - Non Eating(spoon)')
        plt.ylabel('Orientation FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Orientation FFT - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        

        if self.__show_plot == True:
            plt.show() 

    def plot_acceleration_fft(self):
        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_fork[:,4], label='x')
        plt.plot(self.__fft_eating_fork[:,5], label='y')
        plt.plot(self.__fft_eating_fork[:,6], label='z')
        plt.title('Acceleration FFT - Eating(fork)')
        plt.ylabel('Acceleration FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration FFT - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_fork[:,4], label='x')
        plt.plot(self.__fft_non_eating_fork[:,5], label='y')
        plt.plot(self.__fft_non_eating_fork[:,6], label='z')
        plt.title('Acceleration FFT - Non Eating(fork)')
        plt.ylabel('Acceleration FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration FFT - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)


        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_spoon[:,4], label='x')
        plt.plot(self.__fft_eating_spoon[:,5], label='y')
        plt.plot(self.__fft_eating_spoon[:,6], label='z')
        plt.title('Acceleration FFT - Eating(spoon)')
        plt.ylabel('Acceleration FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Acceleration FFT - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_spoon[:,4], label='x')
        plt.plot(self.__fft_non_eating_spoon[:,5], label='y')
        plt.plot(self.__fft_non_eating_spoon[:,6], label='z')
        plt.title('Acceleration FFT - Non Eating(spoon)')
        plt.ylabel('Acceleration FFT')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration FFT - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        if self.__show_plot == True:
            plt.show() 

    def plot_gyro_fft(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_fork[:,7], label='x')
        plt.plot(self.__fft_eating_fork[:,8], label='y')
        plt.plot(self.__fft_eating_fork[:,9], label='z')
        plt.title('Gyroscope FFT - Eating(fork)')
        plt.ylabel('Gyroscope FFT')
        plt.xlabel('Sample number')        
        plt.legend()        
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope FFT - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_fork[:,7], label='x')
        plt.plot(self.__fft_non_eating_fork[:,8], label='y')
        plt.plot(self.__fft_non_eating_fork[:,9], label='z')
        plt.title('Gyroscope FFT - Non Eating(fork)')
        plt.ylabel('Gyroscope FFT')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope FFT - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_eating_spoon[:,7], label='x')
        plt.plot(self.__fft_eating_spoon[:,8], label='y')
        plt.plot(self.__fft_eating_spoon[:,9], label='z')
        plt.title('Gyroscope FFT - Eating(spoon)')
        plt.ylabel('Gyroscope FFT')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope FFT - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__fft_non_eating_spoon[:,7], label='x')
        plt.plot(self.__fft_non_eating_spoon[:,8], label='y')
        plt.plot(self.__fft_non_eating_spoon[:,9], label='z')
        plt.title('Gyroscope FFT - Non Eating(spoon)')
        plt.ylabel('Gyroscope FFT')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope FFT - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 
        




################







    def plot_gyro_mean(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_fork[:,7], label='x')
        plt.plot(self.__mean_eating_fork[:,8], label='y')
        plt.plot(self.__mean_eating_fork[:,9], label='z')
        plt.title('Gyroscope Mean - Eating(fork)')
        plt.ylabel('Gyroscope Mean')
        plt.xlabel('Sample number')        
        plt.legend()        
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope Mean - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_fork[:,7], label='x')
        plt.plot(self.__mean_non_eating_fork[:,8], label='y')
        plt.plot(self.__mean_non_eating_fork[:,9], label='z')
        plt.title('Gyroscope Mean - Non Eating(fork)')
        plt.ylabel('Gyroscope Mean')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope Mean - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_spoon[:,7], label='x')
        plt.plot(self.__mean_eating_spoon[:,8], label='y')
        plt.plot(self.__mean_eating_spoon[:,9], label='z')
        plt.title('Gyroscope Mean - Eating(spoon)')
        plt.ylabel('Gyroscope Mean')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope Mean - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_spoon[:,7], label='x')
        plt.plot(self.__mean_non_eating_spoon[:,8], label='y')
        plt.plot(self.__mean_non_eating_spoon[:,9], label='z')
        plt.title('Gyroscope Mean - Non Eating(spoon)')
        plt.ylabel('Gyroscope Mean')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope Mean - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 
        

    def plot_gyro_variance(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_eating_fork[:,7], label='x')
        plt.plot(self.__variance_eating_fork[:,8], label='y')
        plt.plot(self.__variance_eating_fork[:,9], label='z')
        plt.title('Gyroscope Variance - Eating(fork)')
        plt.ylabel('Gyroscope Variance')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope Variance - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_fork[:,7], label='x')
        plt.plot(self.__variance_non_eating_fork[:,8], label='y')
        plt.plot(self.__variance_non_eating_fork[:,9], label='z')
        plt.title('Gyroscope Variance - Non Eating(fork)')
        plt.ylabel('Gyroscope Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope Variance - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)

        plt.figure(self.get_number())
        plt.plot(self.__variance_eating_spoon[:,7], label='x')
        plt.plot(self.__variance_eating_spoon[:,8], label='y')
        plt.plot(self.__variance_eating_spoon[:,9], label='z')
        plt.title('Gyroscope Variance - Eating(spoon)')
        plt.ylabel('Gyroscope Variance')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope Variance - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_spoon[:,7], label='x')
        plt.plot(self.__variance_non_eating_spoon[:,8], label='y')
        plt.plot(self.__variance_non_eating_spoon[:,9], label='z')
        plt.title('Gyroscope Variance - Non Eating(spoon)')
        plt.ylabel('Gyroscope Variance')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope Variance - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show()           


    def plot_gyro_rms(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__rms_eating_fork[:,7], label='x')
        plt.plot(self.__rms_eating_fork[:,8], label='y')
        plt.plot(self.__rms_eating_fork[:,9], label='z')
        plt.title('Gyroscope RMS - Eating(fork)')
        plt.ylabel('Gyroscope RMS')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope RMS - Eating(fork)'+".png")
        plt.close(self.__figure_number)

        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_fork[:,7], label='x')
        plt.plot(self.__rms_non_eating_fork[:,8], label='y')
        plt.plot(self.__rms_non_eating_fork[:,9], label='z')
        plt.title('Gyroscope RMS - Non Eating(fork)')
        plt.ylabel('Gyroscope RMS')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Gyroscope RMS - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)

        plt.figure(self.get_number())
        plt.plot(self.__rms_eating_spoon[:,7], label='x')
        plt.plot(self.__rms_eating_spoon[:,8], label='y')
        plt.plot(self.__rms_eating_spoon[:,9], label='z')
        plt.title('Gyroscope RMS - Eating(spoon)')
        plt.ylabel('Gyroscope RMS')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope RMS - Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_spoon[:,7], label='x')
        plt.plot(self.__rms_non_eating_spoon[:,8], label='y')
        plt.plot(self.__rms_non_eating_spoon[:,9], label='z')
        plt.title('Gyroscope RMS - Non Eating(spoon)')
        plt.ylabel('Gyroscope RMS')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Gyroscope RMS - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 


    def plot_acceleration_mean(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_fork[:,4], label='x')
        plt.plot(self.__mean_eating_fork[:,5], label='y')
        plt.plot(self.__mean_eating_fork[:,6], label='z')
        plt.title('Acceleration Mean - Eating(fork)')
        plt.ylabel('Acceleration Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Mean - Eating(fork)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_fork[:,4], label='x')
        plt.plot(self.__mean_non_eating_fork[:,5], label='y')
        plt.plot(self.__mean_non_eating_fork[:,6], label='z')
        plt.title('Acceleration Mean - Non Eating(fork)')
        plt.ylabel('Acceleration Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Mean - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)


        plt.figure(self.get_number())
        plt.plot(self.__mean_eating_spoon[:,4], label='x')
        plt.plot(self.__mean_eating_spoon[:,5], label='y')
        plt.plot(self.__mean_eating_spoon[:,6], label='z')
        plt.title('Acceleration Mean - Eating(spoon)')
        plt.ylabel('Acceleration Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Acceleration Mean - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__mean_non_eating_spoon[:,4], label='x')
        plt.plot(self.__mean_non_eating_spoon[:,5], label='y')
        plt.plot(self.__mean_non_eating_spoon[:,6], label='z')
        plt.title('Acceleration Mean - Non Eating(spoon)')
        plt.ylabel('Acceleration Mean')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Mean - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        if self.__show_plot == True:
            plt.show() 


    def plot_acceleration_variance(self):
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_eating_fork[:,4], label='x')
        plt.plot(self.__variance_eating_fork[:,5], label='y')
        plt.plot(self.__variance_eating_fork[:,6], label='z')
        plt.title('Acceleration Variance - Eating(fork)')
        plt.ylabel('Acceleration Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Variance - Eating(fork)'+".png")
        plt.close(self.__figure_number)
                
        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_fork[:,4], label='x')
        plt.plot(self.__variance_non_eating_fork[:,5], label='y')
        plt.plot(self.__variance_non_eating_fork[:,6], label='z')
        plt.title('Acceleration Variance - Non Eating(fork)')
        plt.ylabel('Acceleration Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number)+ '_' + 'Acceleration Variance - Non Eating(fork)'+".png")
        plt.close(self.__figure_number)


        plt.figure(self.get_number())        
        plt.plot(self.__variance_eating_spoon[:,4], label='x')
        plt.plot(self.__variance_eating_spoon[:,5], label='y')
        plt.plot(self.__variance_eating_spoon[:,6], label='z')
        plt.title('Acceleration Variance - Eating(spoon)')
        plt.ylabel('Acceleration Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Variance - Eating(spoon)'+".png")
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__variance_non_eating_spoon[:,4], label='x')
        plt.plot(self.__variance_non_eating_spoon[:,5], label='y')
        plt.plot(self.__variance_non_eating_spoon[:,6], label='z')
        plt.title('Acceleration Variance - Non Eating(spoon)')
        plt.ylabel('Acceleration Variance')
        plt.xlabel('Sample number')
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration Variance - Non Eating(spoon)'+".png")
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 


    def plot_acceleration_rms(self):        
        plt.figure(self.get_number())
        plt.plot(self.__rms_eating_fork[:,4], label='x')
        plt.plot(self.__rms_eating_fork[:,5], label='y')
        plt.plot(self.__rms_eating_fork[:,6], label='z')
        plt.title('Acceleration RMS - Eating(fork)')
        plt.ylabel('Acceleration RMS')
        plt.xlabel('Sample number') 
        plt.legend()       
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration RMS - Eating(fork)'+".png")
        plt.close(self.__figure_number)
                
        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_fork[:,4], label='x')
        plt.plot(self.__rms_non_eating_fork[:,5], label='y')
        plt.plot(self.__rms_non_eating_fork[:,6], label='z')
        plt.title('Acceleration RMS - Non Eating(fork)')
        plt.ylabel('Acceleration RMS')
        plt.xlabel('Sample number')     
        plt.legend()  
        plt.savefig(str(self.__figure_number)+ '_' + 'Acceleration RMS - Non Eating(fork)'+".png")
        
        plt.close(self.__figure_number)

        plt.figure(self.get_number())        
        plt.plot(self.__rms_eating_spoon[:,4], label='x')
        plt.plot(self.__rms_eating_spoon[:,5], label='y')
        plt.plot(self.__rms_eating_spoon[:,6], label='z')
        plt.title('Acceleration RMS - Eating(spoon)')
        plt.ylabel('Acceleration RMS')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration RMS - Eating(spoon)'+".png")
        
        plt.close(self.__figure_number)
        
        plt.figure(self.get_number())
        plt.plot(self.__rms_non_eating_spoon[:,4], label='x')
        plt.plot(self.__rms_non_eating_spoon[:,5], label='y')
        plt.plot(self.__rms_non_eating_spoon[:,6], label='z')
        plt.title('Acceleration RMS - Non Eating(spoon)')
        plt.ylabel('Acceleration RMS')
        plt.xlabel('Sample number')        
        plt.legend()
        plt.savefig(str(self.__figure_number) + '_' + 'Acceleration RMS - Non Eating(spoon)'+".png")
        
        plt.close(self.__figure_number)

        if self.__show_plot == True:
            plt.show() 


    def acceleration_mean(self):
        """
        Calculate the mean acceleration for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        mean acceleration 
        """
        print("\n###########################################################################")
        print("Calculate acceleration mean")
        print("###########################################################################")

        acceleration_mean_fork_eating = np.zeros([len(self.__imu_fork_eating),4],dtype=float)
        acceleration_mean_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),4],dtype=float) 
        acceleration_mean_spoon_eating = np.zeros([len(self.__imu_spoon_eating),4],dtype=float)
        acceleration_mean_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),4],dtype=float)

        x = self.__accelerometer_axis.get("x")
        y = self.__accelerometer_axis.get("y")
        z = self.__accelerometer_axis.get("z")

        # calculate the mean acceleration on the x,y,z axis (columns)        
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            acc_mean = np.mean(user_fork_eating,axis = 0)
            acceleration_mean_fork_eating[i][0] = acc_mean[x]
            acceleration_mean_fork_eating[i][1] = acc_mean[y]
            acceleration_mean_fork_eating[i][2] = acc_mean[z]
            
            user_fork_non_eating = self.__imu_fork_non_eating[i]            
            acc_mean = np.mean(user_fork_non_eating,axis = 0)          
            acceleration_mean_non_eating_fork[i][0] = acc_mean[x]
            acceleration_mean_non_eating_fork[i][1] = acc_mean[y]
            acceleration_mean_non_eating_fork[i][2] = acc_mean[z]

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]            
            acc_mean = np.mean(user_eating,axis = 0)
            acceleration_mean_spoon_eating[i][0] = acc_mean[x]
            acceleration_mean_spoon_eating[i][1] = acc_mean[y]
            acceleration_mean_spoon_eating[i][2] = acc_mean[z]

            user_non_eating = self.__imu_spoon_non_eating[i]            
            acc_mean = np.mean(user_non_eating,axis = 0)          
            acceleration_mean_non_eating_spoon[i][0] = acc_mean[x]
            acceleration_mean_non_eating_spoon[i][1] = acc_mean[y]
            acceleration_mean_non_eating_spoon[i][2] = acc_mean[z]

        self.__acceleration_mean_eating_fork = acceleration_mean_fork_eating
        self.__acceleration_mean_non_eating_fork = acceleration_mean_non_eating_fork
        self.__acceleration_mean_eating_spoon = acceleration_mean_spoon_eating
        self.__acceleration_mean_non_eating_spoon = acceleration_mean_non_eating_spoon
        
        if self.__plot == True:
            self.plot_acceleration_mean()
        

    def acceleration_variance(self):
        """
        Calculate the variance acceleration for axis x,y,z
        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("###########################################################################")
        print("Calculate acceleration variance")
        print("###########################################################################")
        acceleration_variance_fork_eating = np.zeros([len(self.__imu_fork_eating),4],dtype=float)
        acceleration_variance_non_eating_fork = np.zeros([len(self.__imu_fork_non_eating),4],dtype=float) 
        acceleration_variance_spoon_eating = np.zeros([len(self.__imu_spoon_eating),4],dtype=float)
        acceleration_variance_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),4],dtype=float)


        x = self.__accelerometer_axis.get("x")
        y = self.__accelerometer_axis.get("y")
        z = self.__accelerometer_axis.get("z")

        # calculate the mean acceleration on the x,y,z axis (columns)        
        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            acc_variance = np.var(user_fork_eating,axis = 0)
            acceleration_variance_fork_eating[i][0] = acc_variance[x]
            acceleration_variance_fork_eating[i][1] = acc_variance[y]
            acceleration_variance_fork_eating[i][2] = acc_variance[z]
            
            user_fork_non_eating = self.__imu_fork_non_eating[i]            
            acc_variance = np.var(user_fork_non_eating,axis = 0)          
            acceleration_variance_non_eating_fork[i][0] = acc_variance[x]
            acceleration_variance_non_eating_fork[i][1] = acc_variance[y]
            acceleration_variance_non_eating_fork[i][2] = acc_variance[z]

        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_spoon_eating)):
            user_eating = self.__imu_spoon_eating[i]            
            acc_variance = np.var(user_eating,axis = 0)
            acceleration_variance_spoon_eating[i][0] = acc_variance[x]
            acceleration_variance_spoon_eating[i][1] = acc_variance[y]
            acceleration_variance_spoon_eating[i][2] = acc_variance[z]

            user_non_eating = self.__imu_spoon_non_eating[i]            
            acc_variance = np.var(user_non_eating,axis = 0)          
            acceleration_variance_non_eating_spoon[i][0] = acc_variance[x]
            acceleration_variance_non_eating_spoon[i][1] = acc_variance[y]
            acceleration_variance_non_eating_spoon[i][2] = acc_variance[z]

        self.__acceleration_variance_eating_fork = acceleration_variance_fork_eating
        self.__acceleration_variance_non_eating_fork = acceleration_variance_non_eating_fork
        self.__acceleration_variance_eating_spoon= acceleration_variance_spoon_eating
        self.__acceleration_variance_non_eating_spoon = acceleration_variance_non_eating_spoon
        
        if self.__plot == True:
            self.plot_acceleration_variance() 



    def acceleration_energy(self):
        """
        Calculate the acceleration energy for each axis
        Parameters
        ----------
        None

        Return
        ------
        None
        """        
        print("###########################################################################")
        print("Calculate acceleration energy")
        print("###########################################################################")        

    def energy_spectral_density(self):
        """
        Feature extraction step
        Parameters
        ----------
        None

        Return
        ------
        None
        """    
        print("###########################################################################")
        print("Calculate spectral density")
        print("###########################################################################")                
    


def main():        

    activityRecognition = ActivityRecognition()
    activityRecognition.preprocessing()
    activityRecognition.feature_extraction()
    #activityRecognition.pca()

if __name__ == "__main__":
    main()
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
        self.__plot = False
        self.__save = False
        self.__log_enabled = False                        
        self.__ground_truth_file_list = []

        self.__all_imu_myo_fork_data = []
        self.__imu_myo_fork_data_userid = []
        self.__all_imu_myo_spoon_data = []
        self.__imu_myo_spoon_data_userid = []
        self.__imu_fork_eating = []
        self.__imu_fork_non_eating = []
        self.__imu_spoon_eating = []
        self.__imu_spoon_non_eating = []

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

        
        #temp1 = self.__all_imu_myo_spoon_data[0]
        #plt.figure(1)
        #plt.plot(temp1[:,0])
        #plt.plot(temp1[:,1])
        #plt.plot(temp1[:,2])
        #plt.ylabel('orintation')
        #
        #plt.figure(2)
        #plt.plot(temp1[:,4])
        #plt.plot(temp1[:,5])
        #plt.plot(temp1[:,6])
        #plt.ylabel('acceleration')
        #plt.show()

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
            
            # save to file the non eating with fork IMU data
            np.savetxt(user_dir_path_fork + non_eating_file_prefix + '.csv',extracted_imu_data, delimiter=',')

            # make sure to delete the first row of ones
            temp_fork_eating = np.delete(temp_fork_eating,0,0)

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
        self.acceleration_mean()
        self.acceleration_variance()
        self.acceleration_energy()
        self.energy_spectral_density()


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
        # iterate the eating and non eating data for all the users
        # mean_x, mean_y, mean_z
        # preprocessing method is a prerequisit for this step
        #self.__imu_fork_eating = []
        #self.__imu_spoon_eating = [] 
        #self.__imu_fork_non_eating = []
        #self.__imu_spoon_non_eating = []
        
        # x,y,z
        acceleration_mean_eating = np.zeros([len(self.__imu_fork_eating),3],dtype=float)
        acceleration_mean_non_eating = np.zeros([len(self.__imu_fork_non_eating),3],dtype=float)
        
        #acceleration_mean_spoon_eating = np.zeros([len(self.__imu_spoon_eating),3],dtype=float)
        #acceleration_mean_non_eating_spoon = np.zeros([len(self.__imu_spoon_non_eating),3],dtype=float)
        

        for i in range(len(self.__imu_fork_eating)):
            user_fork_eating = self.__imu_fork_eating[i]
            #user_spoon_eating = self.__imu_spoon_eating[i]
            bla = np.mean(user_fork_eating,axis = 0)

            acceleration_mean_eating[i][0] = bla[4]
            acceleration_mean_eating[i][1] = bla[5]
            acceleration_mean_eating[i][2] = bla[6]
            #acceleration_mean_eating[i][0] = np.mean(user_fork_eating[:,4])
            #acceleration_mean_eating[i][1] = np.mean(user_fork_eating[:,5])
            #acceleration_mean_eating[i][2] = np.mean(user_fork_eating[:,6])

        # take the same number of rows
        for i in range(len(self.__imu_fork_eating)):
            user_fork_non_eating = self.__imu_fork_non_eating[i]
            bla = np.mean(user_fork_non_eating,axis = 0)          
            acceleration_mean_non_eating[i][0] = bla[4]
            acceleration_mean_non_eating[i][1] = bla[5]
            acceleration_mean_non_eating[i][2] = bla[6]
        
                
    
        plt.figure(1)
        plt.plot(acceleration_mean_eating[:,0])
        plt.plot(acceleration_mean_eating[:,1])
        plt.plot(acceleration_mean_eating[:,2])
        plt.ylabel('Mean eating - acceleration')
        #
        plt.figure(2)
        plt.plot(acceleration_mean_non_eating[:,0])
        plt.plot(acceleration_mean_non_eating[:,1])
        plt.plot(acceleration_mean_non_eating[:,2])
        plt.ylabel('Mean non eating - acceleration')
        plt.show()
    
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
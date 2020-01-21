import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from tempfile import TemporaryFile
import re


## iterate the ground truth
## per each user
## get the start and end
## for the IMU
## Eatning 
## start*50/30
## end*50/30
## Non eating - all the rest
## 
class ActivityRecognition:
    def __init__(self, ground_truth_dir_path=None, myo_data_dir_path=None):
        self.__data_dir_path = ground_truth_dir_path 
        self.__data_file_list = ["AllSamples.mat"]
        self.__plot = False
        self.__save = False
        self.__log_enabled = False                                         
        # there are gaps in user ids. User 15 does not exsists
        self.__start_user_id = 9
        self.__end_user_id = 41
        self.__ground_truth_file_list = []
        
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
        
        #print ("Ground truth")
        #for i in range(len(self.__ground_truth_file_list)):
        #    print self.__ground_truth_file_list[i]
        #print ("My Data")
        #for i in range(len(self.__myo_data_file_list)):
        #    print(self.__myo_data_file_list)[i]
        
        


    def cleanup(self):
        print ("Cleanup")


    def data_setup(self):

        # all should be class members
        # should be defined in the constructor
        all_imu_myo_fork_data = []
        imu_myo_fork_data_userid = []
        all_imu_myo_spoon_data = []
        imu_myo_spoon_data_userid = []
        imu_fork_eating = []
        imu_fork_non_eating = []
        imu_spoon_eating = []
        imu_spoon_non_eating = []
        

        for i in range (len(self.__myo_data_file_list)):
            file = self.__myo_data_file_list[i]
            if "IMU" in file:
                if "fork" in file:
                    result = re.search('MyoData/(.*)/fork', file)
                    imu_myo_fork_data_userid.append(result.group(1))
                    imu_myo_fork_data=np.genfromtxt(file, dtype=float, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10))
                    # Add another column of zeros representing non eating action     
                    all_imu_myo_fork_data.append(imu_myo_fork_data)
                elif "spoon" in file:
                    result = re.search('MyoData/(.*)/spoon', file)
                    imu_myo_spoon_data_userid.append(result.group(1))
                    imu_myo_spoon_data=np.genfromtxt(file, dtype=float, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10))
                    all_imu_myo_spoon_data.append(imu_myo_spoon_data)
                else:
                    print "unknown file.continue"  

                
        print("IMU MYO SPOON DATA USERS")
        print(imu_myo_spoon_data_userid)
        print("IMU MYO FORK DATA USERS")
        print(imu_myo_spoon_data_userid)

        
        # fork eating, fork non eating
        # spoon eating, spoon non eating

        # Fork
        all_ground_truth_fork_data = []
        ground_truth_fork_data_userid = []
        # Spoon
        all_ground_truth_spoon_data = []
        ground_truth_spoon_data_userid = []

        # Iterate all ground truth data
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
                print "unknown file.continue"


        print("GROUND TRUTH SPOON USERS")
        print(ground_truth_spoon_data_userid)
        print("GROUND TRUTH FORK USERS")
        print(ground_truth_fork_data_userid)

        
        # start with fork
        for i in range (len(all_ground_truth_fork_data)):
            fork_data = all_ground_truth_fork_data[i]
            userid = ground_truth_fork_data_userid[i]
            print(userid)
            #  get the index for this user id
            for k in range (np.size(imu_myo_fork_data_userid)):
                if (userid == imu_myo_fork_data_userid[k]):
                    uid = k
                    print("FOUND, ",k)
                    break
            
            myo_fork_data = all_imu_myo_fork_data[k]
            print(type(myo_fork_data))
            print(myo_fork_data.shape)
            # create a temp numpy array of zeros
            eating_action_range = np.zeros([len(fork_data),2], dtype=int)
            # should get the size according to the shape
            temp_fork_eating = np.ones((1,10),dtype=float)
            print(temp_fork_eating)
            temp_fork_non_eating = np.ones((1,10))

            for j in range(len(fork_data)):
                start_frame = fork_data[j][0]
                end_frame = fork_data[j][1]
                print ("Start = ", start_frame)
                print ("End = ", end_frame)
                start_row = (start_frame * 50)/30
                end_row = (end_frame * 50)/30
                eating_action_range[j][0] = start_row
                eating_action_range[j][1] = end_row
                cc = myo_fork_data[start_row:end_row]
                print("*****CC**")
                print(type(cc))
                print(cc.shape)
                #print(cc)
                print("*****TEMP**")
                print(type(temp_fork_eating))
                print(temp_fork_eating.shape)
                #print(temp_fork_eating)
                
                temp_fork_eating = np.concatenate((temp_fork_eating,cc))
                print("*****CONCAT**")
                print(type(temp_fork_eating))
                print(temp_fork_eating.shape)
                print("*****END**")

            
            imu_fork_eating.append(temp_fork_eating)
            temp_fork_eating = []


        np.savetxt('data.csv',imu_fork_eating[2],delimiter=',')


    def load_data(self):
        """
        Load the input data (e.g. the samples to be clusters)
        Parameters
        ----------
        None

        Return
        ------
        None     
        """
        self.__data = scipy.io.loadmat (self.__data_path + "/AllSamples.mat")
        self.__unlabeled_data = self.__data['AllSamples']

        if (self.__plot == True):
            colors = (0,0,0)
            area = np.pi*3
            self.__X1 = self.__unlabeled_data[:,[0]]
            self.__X2 = self.__unlabeled_data[:,[1]]
            plt.scatter(self.__X1, self.__X2,s=area,c=colors,alpha=0.5)
            plt.show()
            plt.title('Unlabled data scatter plot')
            plt.xlabel('X1')
            plt.ylabel('X2')
        
        if (self.__save == False):
            np.savetxt("data", self.__unlabeled_data,delimiter=",")
        



    


def main():

    #objective_function_1 = []
    #objective_function_2 = []

    activityRecognition = ActivityRecognition()

    activityRecognition.data_setup()
    #number_of_clusters = 10
    #number_of_runs = 2
    #k = np.arange(2,number_of_clusters+1)
#
    #plt.title('Elbow Graph')
    #plt.xlabel('Number of Clusters K')
    #plt.ylabel('Objective Function Value')
#
    #for run in range(1, number_of_runs+1):
    #    centroid_init_strategy = 1
    #    for _k in range(2,number_of_clusters+1):
    #        ActivityRecognition.setup(centroid_init_strategy,_k)
    #        objective_function_1.append(ActivityRecognition.compute())
    #    label = "Strategy 1 - run" + str(run)
    #    plt.plot (k,objective_function_1, label = label)
    #    plt.legend()
    #    #plt.show()
#
    #    centroid_init_strategy = 2
    #    for _k in range(2,number_of_clusters+1):
    #        ActivityRecognition.setup(centroid_init_strategy ,_k)
    #        objective_function_2.append(ActivityRecognition.compute())
    #    label = "Strategy 2 - run" + str(run)
    #    plt.plot (k,objective_function_2, label = label)
    #    plt.legend()
    #    #plt.show()
    #    
    #    objective_function_1 = []
    #    objective_function_2 = []
    #
#
    #plt.show()
#
    

if __name__ == "__main__":
    main()
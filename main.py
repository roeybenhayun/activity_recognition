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
# imports for Project2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neural_network import MLPClassifier

class ActivityRecognition:
    def __init__(self, ground_truth_dir_path=None, myo_data_dir_path=None):
        self.__data_dir_path = ground_truth_dir_path
        self.__plot = False
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

        self.__imu_eating = []
        self.__imu_non_eating = []

        self.__X = []

        self.__number_of_rows_per_user = []
        # MEAN
        self.__mean_eating = []
        self.__mean_non_eating = []
        
        # VAR
        self.__variance_eating = []
        self.__variance_non_eating = []
                
        # RMS
        self.__rms_eating = []
        self.__rms_non_eating = []
        

        # MAX
        self.__min_eating = []
        self.__min_non_eating = []
        

        # MIN
        self.__max_eating = []
        self.__max_non_eating = []
        

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

        
        # only if this flag enabled than there is a point in checking whether dir exists
        if self.__save_files == True:
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


        if self.__save_files == True:
            self.create_plot_dir()
                
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
                    if self.__save_files == True:
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
            
            # Roey - that's enough to have only eating and non-eating activities (no need to seperate for spoon and fork)
            self.__imu_fork_eating.append(temp_fork_eating)
            #self.__imu_eating.append(temp_fork_eating)
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
                    if self.__save_files == True:
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
            if self.__save_files == True:
                np.savetxt(user_dir_path_spoon + non_eating_file_prefix + '.csv',extracted_imu_data, delimiter=',')


            # make sure to delete the first row of ones
            temp_spoon_eating = np.delete(temp_spoon_eating,0,0)

            # Save the IMU data for fork eating
            if self.__save_files == True:
                np.savetxt(user_dir_path_spoon + file_prefix + '.csv',temp_spoon_eating, delimiter=',')
            
            self.__imu_spoon_eating.append(temp_spoon_eating)

            temp_spoon_eating = []

            # At this point we have spliited the data into two list:
            # eating and non-eating for each user
        
        
        ##### Combine eating-fork and eating-spoon to eating activity
        ##### Combine non-eating-fork and non-eating-spoon to non-eating activity
        for i in range(len(self.__imu_spoon_eating)):
            eating_spoon = self.__imu_spoon_eating[i]
            eating_fork = self.__imu_fork_eating[i]
            eating = np.vstack((eating_spoon,eating_fork))
            print("EATING : ", np.shape(eating))
            self.__imu_eating.append(eating)

            non_eating_spoon = self.__imu_spoon_non_eating[i]
            non_eating_fork = self.__imu_fork_non_eating[i]
            non_eating = np.vstack((non_eating_spoon,non_eating_fork))

            # the non-eating is a sub set of the eating
            # set the same number of rows for eating and non eating
            non_eating = np.delete(non_eating,slice(len(eating),len(non_eating)),axis=0)
        
            print("NON EATING" , np.shape(non_eating))
            self.__imu_non_eating.append(non_eating)

        print("***********************")
        print(len(self.__imu_eating))
        print(len(self.__imu_non_eating))
        print("***********************")


        
    def feature_extraction(self):
        print("\n")            
        print("###########################################################################")
        print("Feature extraction: Calculate Mean of eating and non eating...")
        print("###########################################################################")
        
        eating = np.zeros([(len(self.__imu_eating)),10],dtype=float)
        non_eating = np.zeros([(len(self.__imu_non_eating)),10],dtype=float) 

        window_size = 20
        # calculate the mean acceleration on the x,y,z axis (columns)
        for i in range(len(self.__imu_eating)):
            # get the eatig and non eating samples
            eating = self.__imu_eating[i]
            non_eating = self.__imu_non_eating[i]

            # get the divide and reminder
            quotient = len(eating)//window_size
            #print("QQ = ",quotient)
            
            # this will be used later for splitting the user data
            self.__number_of_rows_per_user.append(quotient)

            reminder = np.mod(len(eating),window_size)
            
            
            start=0

            for j in range(0,quotient):
                end = start + window_size
                #print("START = ",start,"END = ",end)

                # extract all features - eating
                # Mean
                self.__mean_eating.append(np.mean(eating[start:end],axis = 0))                
                # Variance
                self.__variance_eating.append(np.var(eating[start:end],axis = 0))
                # Min
                self.__min_eating.append(np.min(eating[start:end],axis = 0))
                # Max
                self.__max_eating.append(np.max(eating[start:end],axis = 0))
                # RMS
                self.__rms_eating.append(np.sqrt(np.mean(eating[start:end] ** 2,axis = 0)))
        
                # extract all features - non eating
                # Mean
                self.__mean_non_eating.append(np.mean(non_eating[start:end],axis = 0))
                # Variance
                self.__variance_non_eating.append(np.var(non_eating[start:end],axis = 0))
                # Min
                self.__min_non_eating.append(np.min(non_eating[start:end],axis = 0))
                # Max
                self.__max_non_eating.append(np.max(non_eating[start:end],axis = 0))
                # RMS
                self.__rms_non_eating .append(np.sqrt(np.mean(non_eating[start:end] ** 2,axis = 0)))
                
                start=end


        #print("LEN mean eating = ",len(self.__mean_eating))
        #print("LEN variance eating = ", len(self.__variance_eating))
        #print("LEN rms eating = ", len(self.__rms_eating))
        #print("LEN min eating = ", len(self.__min_eating))
        #print("LEN max eating = ", len(self.__max_eating))
        #
#
        #print("LEN mean non eating = ",len(self.__mean_non_eating))
        #print("LEN variance non eating = ", len(self.__variance_non_eating))
        #print("LEN rms non eating = ", len(self.__rms_non_eating))
        #print("LEN min non eating = ", len(self.__min_non_eating))
        #print("LEN max non eating = ", len(self.__max_non_eating))

        X_1 = np.hstack\
            ((self.__mean_eating,\
            self.__variance_eating, \
            self.__rms_eating, \
            self.__min_eating, \
            self.__max_eating))
        
        X_2 = np.hstack\
            ((self.__mean_non_eating,\
            self.__variance_non_eating, \
            self.__rms_non_eating, \
            self.__min_non_eating, \
            self.__max_non_eating))
        

        # This the feature matrix which will be the input to PCA
        self.__X = np.vstack((X_1,X_2))

        #print(np.shape(X))
                

    def pca(self):
        print("\n")
        print("###########################################################################")
        print("PCA")
        print("###########################################################################")
        X_reduced = PCA(n_components=5).fit_transform(self.__X)
        return X_reduced, self.__number_of_rows_per_user


class UserDependentAnalysis:
    def __init__(self,X_reduced=None,user_data_info_list=None):
        print("User Dependent Analysis constructor")
        self.__X_reduced = X_reduced
        self.__user_data_info_list = user_data_info_list
        


    def split_dataset(self,split_type):
        print("\n")
        print("###########################################################################")
        print("Data set split")
        print("###########################################################################")
        
        if (split_type == 'user_data_60_40'):
            print(self.__user_data_info_list)
            # split 60% for training and 40% for testing
            start = 0
            for i in range(0,len(self.__user_data_info_list)):            
                user_data = self.__user_data_info_list[i]
                user_data_testing_range = int(np.round(user_data*0.6))
                print (user_data_testing_range)
                print("START = ", start, "END = ",user_data_testing_range)
                
                end = user_data - user_data_testing_range
                training = self.__X_reduced[start:start+user_data_testing_range,:]
                testing = self.__X_reduced[start+user_data_testing_range:start+user_data_testing_range+end,:]

                print(np.shape(training))
                print(np.shape(testing))
                
                start = start + user_data

                
                #X_train.append(self.__X_reduced(start:user_data_testing_range))

        elif (split_type == 'user_id_60_40'):
            #y[0:60]='y'
            #y[60:119] ='n'
            Y = []
            for i in range(60):
                Y.append("eating")
            for i in range(60,120):
                Y.append("non_eating")

            # Split the data in 60% training/validation and 40% test set. random_state was set to a constant value
            # in order to get conssistent results when we re run the code
            X_train, X_test, Y_train, Y_test = train_test_split(self.__X_reduced , Y, test_size=0.4, random_state=42)

            print(np.shape(self.__X_reduced ))
            print("train test spllit")
            print("X_train shape: " ,np.shape(X_train))
            print("X_test shape : " ,np.shape(X_test))
            print("Y_train shape : " ,np.shape(Y_train))
            print("Y_test shape : " ,np.shape(Y_test))
        
        else:
            print("Invalid split type")

        
        return 0,0,0,0
        #return X_train,X_test,Y_train,Y_test
    
    def classify(self,classifier,X_train, X_test, Y_train, Y_test):
        if (classifier == 'svm'):
            print("SVM Classifier")
            clf = LinearSVC(penalty='l2', loss='squared_hinge',
                    dual=True, tol=0.0001, C=100, multi_class='ovr',
                    fit_intercept=True, intercept_scaling=1, class_weight=None,verbose=0
                    , random_state=0, max_iter=1000)
        elif (classifier == 'decision_tree'):
            print("Decision Tree Classifier")
            clf = tree.DecisionTreeClassifier()
        elif (classifier == 'neural_nets'):
            print("Neural Nets Classifier")
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

        else:
            print("Unsupported classifier")

        clf.fit(X_train,Y_train)
        predicted = clf.predict(X_test)
        expected = Y_test

        print( '\nClassification report\n')
        print(classification_report(expected, predicted))



def main():
    # Project 1
    activityRecognition = ActivityRecognition()
    activityRecognition.preprocessing()
    activityRecognition.feature_extraction()
    reduced_feature_matrix,rows_per_user_list = activityRecognition.pca()
    print(np.shape(reduced_feature_matrix))
    

    # Project 2
    userDependentAnalysis = UserDependentAnalysis(reduced_feature_matrix,rows_per_user_list)


    X_train,X_test, Y_train, Y_test = userDependentAnalysis.split_dataset('user_data_60_40')

    #userDependentAnalysis.classify('svm',X_train,X_test, Y_train, Y_test)
    #userDependentAnalysis.classify('decision_tree',X_train,X_test, Y_train, Y_test)
    #userDependentAnalysis.classify('neural_nets',X_train,X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()
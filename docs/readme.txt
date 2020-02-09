Dataset explanation (EMG, IMU) 
IMU file consists of 11 columns: 
UNIX time stamp, 
Orientation X, 
Orientation Y, 
Orientation Z, 
Orientation W, 
Accelerometer X, 
Accelerometer Y, 
Accelerometer Z, 
Gyroscope X, 
Gyroscope Y, 
Gyroscope Z. 
EMG file consists of 9 columns: 
UNIX time stamp, 
EMG 1, 
EMG 2, 
EMG 3, 
EMG 4, 
EMG 5, 
EMG 6, 
EMG 7, 
EMG 8.


Synchronization between Myo and Video
The data collection application does not guarantee what the start time between Myo data and Video frame are the same. 
The end time between Myo and Video, however, is the approximately same. 
Therefore, we can synchronize two dataset by setting the last frame time to the last UNIX time stamp in IMU or EMG file.

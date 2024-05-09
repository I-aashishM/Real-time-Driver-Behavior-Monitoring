# Real-time-Driver-Behavior-Monitoring

## Instructions ##
 ---------------
  ### Setup ###
  #### Colab Setup: ####
  **If you're using Google Colab, mount your Google Drive by executing the provided code cell. This will allow you to access files stored in your Google Drive from Colab.**

****Importing Libraries:**** Run the cell to import necessary Python libraries. Make sure all required libraries are installed; if not, use `pip3 install -r requirements.txt`

- All working scripts are stored in `/ADAMS/Scripts/`
****Note:**** If you're running the project, navigate to `/ADAMS/Scripts/adam.ipynb`


### Training ###
#### 1. Dataset Preparation: ####

- ****Data Collection:**** The project involves collecting data through mobile phones placed behind the car's steering wheel. 
- ****Data Cleaning:**** Follow the steps to convert videos to frames, manually check images for quality, and remove irrelevant or blurry images.
- ****Data Annotation:**** Use of LabelImg for annotating the data for object detection tasks. Our classes are clearly defined and annotated.
- ****Data Augmentation:**** Use of the Albumentations library for augmenting the training data.
- ****Train, Test, Val Split:**** Organize the dataset into training, validation, and test sets as per the given ratio in notebook.

#### 2. Model Training: ####

- Follow the steps outlined for configuring and training the model. Ensure all paths to data directories are correctly set.
- Monitor the training process.

![results](https://github.com/I-aashishM/Real-time-Driver-Behavior-Monitoring/assets/35104828/8b6218fa-844c-4589-87b3-b7e657edb06a)
![P_curve](https://github.com/I-aashishM/Real-time-Driver-Behavior-Monitoring/assets/35104828/b5aa8a42-28c2-4ba4-b19d-8e24e1580a8b)
![R_curve](https://github.com/I-aashishM/Real-time-Driver-Behavior-Monitoring/assets/35104828/36337e36-34ea-4750-b727-b8726777dbc5)



### Inference ###
- ****Model Testing and Evaluation:**** After training, evaluate the model on a test dataset.

![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/69158910-1cb9-4342-9a12-5f4a1335d819)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/150d5499-0f5c-4fe1-86bc-e2620a921b44)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/1978c7f2-2967-44c8-8131-6ab1a4d3dde9)


- ****Model Output on Public Dataset:**** Evaluated a trained model on a publicly available dataset [Driving Monitoring Dataset](https://dmd.vicomtech.org/) from [Roboflow website](https://universe.roboflow.com/drivermonitoring/driver-atention/dataset/4), as mentioned in the notebook, to further validate its performance.

- ****Export:**** Convert trained PyTorch model to ONNX for deployment on edge devices, as mentioned in notebook.

- ****Integration with Downstream Pipeline:****

	- Implement the downstream pipeline for classifying distraction levels and assigning scores to different classes.
	- Grouped classes into super-classes and processing the output accordingly.
   ![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/ceb9552f-b53f-4d87-a05e-38f57d51b7ae)

	- Save the results, like warning status, in a CSV file as outlined.
   
   ![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/86b70e2d-7f9d-45ee-9664-05c2fdfcac04)

- ****References:****
	
 	- Yolov8 from Ultralytics : https://github.com/ultralytics/ultralytics
 	- Driving Monitoring Dataset : Ortega, J., Kose, N., Cañas, P., Chao, M.a., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020). DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision -- ECCV 2020 Workshops (pg. 387–405). Springer International Publishing.
  - Roboflow Dataset : https://universe.roboflow.com/drivermonitoring/driver-atention/dataset/4

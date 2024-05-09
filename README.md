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

  ![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/d44356cd-8bec-4180-8761-96c0a7d696a3)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/9aeb3b6e-df7f-4fa9-8d72-996b0bc8ddd6)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/fa218383-b4f0-4125-8957-2324fc0f8eeb)


### Inference ###
- ****Model Testing and Evaluation:**** After training, evaluate the model on a test dataset.

![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/69158910-1cb9-4342-9a12-5f4a1335d819)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/150d5499-0f5c-4fe1-86bc-e2620a921b44)
![image](https://github.com/UNB-TME-6017-W24/final-project-submission-I-aashishM/assets/35104828/1978c7f2-2967-44c8-8131-6ab1a4d3dde9)


- ****Model Output on Public Dataset:**** Test trained model on a publicly available dataset [Driving Monitoring Dataset](https://dmd.vicomtech.org/) from [Roboflow website](https://universe.roboflow.com/drivermonitoring/driver-atention/dataset/4), as mentioned in the notebook, to further validate its performance.

- ****Export:**** Convert trained PyTorch model to ONNX for deployment on edge devices, as mentioned in notebook and final report.

- ****Integration with Downstream Pipeline:****

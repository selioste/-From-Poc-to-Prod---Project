# Goal

Build a machine learning model that is able to prodict tag given an input test. 

# Folder composition
In the repository, you will a model based on 3 principal folders named:
 - Preprocessing
 - Train
 - Predict

# Roadmap 

To achieve the final result, it was necessary to have a roadmap. This roadmap serves as a guide for the progress of the project. 
We start by working on the preprocessing folder. Once finished, we run it to see if there are any errors. If everything is ok, we move on to the train folder, otherwise we have to check the added lines of code. After preprocessing, the train is also completed and tested. The result obtained will attest to the quality of our model. Finally, at the predict, given a text, we predict tags by means of the implemented model. 


# Useful command

Command to execute the different folder on python:
python -m unittest preprocessing.tests.test_utils
python -m unittest train.tests.test_model_train
python -m unittest predict.tests.test_predict

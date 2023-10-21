This folder should contain the data for stanford cars and icml face data. 

The ICML face data is a CSV that you can download at https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv.

Stanford cars should automatically download, if the URL is not broken. If it 
is, then follow the instructions in this github issue 
https://github.com/pytorch/vision/issues/7545. 

Food 101 is donwloaded through TFDS, so it will end up in the default TFDS 
cache directory. Cifar10 and Cifar100 should also automatically download. 

This folder should look something like this 

### Layout

    Data
    ├── stanford_cars
    │   ├── cars_test
    │   └── cars_train
    └── icml_face_data.csv

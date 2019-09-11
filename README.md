# Link_Prediction_Project
### This is course project for MSBD 5008.
- `data_preprocessing.py` is used to generate training and testing set for xgboost.  
    It will use thost traditional method to calculate score for a nodes pair as features when training.
    Currently the methods used are provided by networkit, may change to our own methods later.

- `classify.py` is used to predict all with those methods and save the results in the `result` folder.

#### Things to do
- compare results for different methods
- friend recommend with xgboost model

#### About LFS
Because github limites the size of one single file to 50M, and our dataset generated has exceeded the limitation,  
LFS is used for storing those large files, which costs me $USD 5 per month (a meal).  
Therefore I will only keep those large files for one month, and then remove them after this semester.  
If you want to keep those data, remember to make a backup in this month.

[About LFS](https://blog.csdn.net/jjjjjj123321/article/details/84890893)  
[Delete previous commits](https://blog.csdn.net/quiet_girl/article/details/79487966)

import os
import numpy as np

location = os.getcwd()

f1 = os.path.join(location,'pos')
f2 = os.path.join(location, 'neg')

def probability(x,review_vocab):
	prob_pos=[]
	prob_neg=[]
	if x == f1:
		arr_pos = np.zeros((1000,9), dtype = 'int')
		
		for dirList, subdirList, fileList in os.walk(x):
			
			for file_loc,file in enumerate(fileList):
				with open(os.path.join(x,file)) as rev_file:
					for word_loc,word in enumerate(review_vocab):
						if(word in text):
							arr_pos[file_loc][word_loc]=1
						
				arr_pos[file_loc][8]=1
		# print(arr_pos)
		prob_pos = np.sum(arr_pos,axis=0)
		prob_pos= prob_pos/1000
		prob_pos=np.delete(prob_pos,8)
		
		

	elif x == f2:
		arr_neg = np.zeros((1000,9), dtype = 'int')
		
		for dirList, subdirList, fileList in os.walk(x):
			
			for file_loc,file in enumerate(fileList):
				with open(os.path.join(x,file)) as rev_file:
					
					#File Text
					text=rev_file.read().lower().split()

					for word_loc,word in enumerate(review_vocab):
						if(word in text):
							arr_neg[file_loc][word_loc]=1
		return arr_neg					
				
		# print(arr_neg)					
		prob_neg = np.sum(arr_neg,axis=0)
		prob_neg= prob_neg/1000
		prob_neg=np.delete(prob_neg,8)
		
        
       
review_vocab = ['awful','bad','boring','dull','effective','enjoyable','great','hilarious']


arr_pos = probability(f1,review_vocab)
arr_neg = probability(f2,review_vocab)
data = []
data = np.concatenate((arr_pos,arr_neg),axis=0) 
shuffle = np.random.permutation(data)   
split_ratio = 0.7
train_size = split_ratio * len(shuffle[:,1]) 
test_size = (1-split_ratio)* len(shuffle[:,1])
train_data = shuffle[:int(train_size)]    # creating training data
test_data = shuffle[int(train_size):]  #creating testing data
pos_word_count=np.sum(train_data[train_data[:,8]==1][:,:-1],axis=0)
neg_word_count=np.sum(train_data[train_data[:,8]==0][:,:-1],axis=0)
prob_tr_pos=pos_word_count/train_size
prob_tr_neg=neg_word_count/train_size

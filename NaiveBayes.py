import numpy as np
import pandas as pd

#### load data ####


train_data=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/train.csv')
test_data=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/test.csv')
word_indices=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/word_indices.txt',names=['word'])
train_data.columns=[word_indices.iloc[i]['word'] for i in word_indices.index]
test_data.columns=[word_indices.iloc[i]['word'] for i in word_indices.index]
train_lable=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/train_labels.txt')
test_lable=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/test_labels.txt')

#### train validation spliting ####

def train_val_split(data,lable,percentage):
    idx=range(len(data))
    idx=list(idx)
    np.random.shuffle(idx)
    aa=int(percentage*len(data)/100)
    idx1=idx[:aa]
    idx2=idx[aa:]
    x_tr=data.iloc[idx1,:]
    y_tr=lable.iloc[idx1]
    x_val=data.iloc[idx2,:]
    y_val=lable.iloc[idx2]
    x_tr.index=range(aa)
    y_tr.index=range(aa)
    x_val.index=range(len(data)-aa)
    y_val.index=range(len(data)-aa)
    return x_tr,y_tr,x_val,y_val    

x_train,y_train,x_val,y_val=train_val_split(train_data,train_lable,80)


#### naive bayes class ####

class NaiveBayes():
    
    def fit(self,x,y):
        self.x=np.array(x)
        self.y=np.array(y)
        self.feat_size=len(self.x[0])
        self.n_sample=len(self.x)
        self.parameters0=[]
        self.parameters1=[]
        idx0 , idx1 = [],[]
        for s in range(self.n_sample):
            if self.y[s]==0: idx0.append(s)
            if self.y[s]==1: idx1.append(s) 
        for feat in range(self.feat_size):
            mean=np.mean(self.x[idx0 , feat])
            var=np.var(self.x[idx0 , feat])
            self.parameters0.append((mean,var))
        for feat in range(self.feat_size):
            mean=np.mean(self.x[idx1 , feat])
            var=np.var(self.x[idx1 , feat])
            self.parameters1.append((mean,var))  
        
    def evaluation(self,x_v,y_v):
        self.x_val=np.array(x_v)
        self.y_val=np.array(y_v)
        def log_likelihood(x,m,v):
            return (np.log((1/(np.sqrt(2*np.pi*v)+0.01))))-((1/((2*v)+0.01))*((x-m)**2))
        y_pred=np.zeros(len(self.x_val))
        for s in range(len(self.x_val)):
            p11=0
            p00=0
            for i,xi in enumerate(self.x_val[s,:]):
                p11+=log_likelihood(xi,self.parameters1[i][0],self.parameters1[i][1])
                p00+=log_likelihood(xi,self.parameters0[i][0],self.parameters0[i][1])
            if p11 >= p00 : y_pred[s]=1
            else : y_pred[s]=0
         
        count=0
        for s in range(len(self.x_val)):
            if y_pred[s]==self.y_val[s]:
                count+=1
        print(f' acc: {count/len(self.y_val)}')


#####
        
NV=NaiveBayes()
NV.fit(x_train,y_train)

NV.evaluation(x_train,y_train)
NV.evaluation(x_val,y_val)



















    
    
import numpy as np
import pandas as pd
import datetime

#### loading data ####

train_data=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/train.csv')
test_data=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/test.csv')
word_indices=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/word_indices.txt',names=['word'])
train_data.columns=[word_indices.iloc[i]['word'] for i in word_indices.index]
test_data.columns=[word_indices.iloc[i]['word'] for i in word_indices.index]
train_lable=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/train_labels.txt')
test_lable=pd.read_csv('C:/Users/NP-soft/Desktop/data/Reuters-21578/test_labels.txt')

#### train validation split function ####

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

x_train,y_train,x_val,y_val=train_val_split(train_data,train_lable,100)

#### k-folf cross validation function ####

def K_Fold(k,x_t,y_t):
    x=np.array(x_t)
    y=np.array(y_t)
    ids_x=list(range(len(x)))
    np.random.shuffle(ids_x)
    sec=int(len(y)/k)
    xx=x[ids_x,:]
    yy=y[ids_x]
    ids_x=list(range(len(x)))
    ids_xx=list(range(len(x)))
    k_fold_val_list=[]
    k_fold_train_list=[]
    k_fold_valy_list=[]
    k_fold_trainy_list=[]
    k_fold_val_ids_list=[]
    k_fold_train_ids_list=[]
    for ki in range(k):
        k_fold_val_ids_list.append(ids_xx[:sec])
        ids_xx=ids_xx[sec:]
    for ki in range(k):
        ids=k_fold_val_ids_list[ki]
        train_ids=list(np.zeros(k))
        for i in range(len(train_ids)):
            train_ids[i]=[]
        for idss in ids_x:
            if idss not in ids: train_ids[ki].append(idss)
        k_fold_train_ids_list.append(train_ids[ki])
        
        k_fold_val_list.append(pd.DataFrame(xx[k_fold_val_ids_list[ki],:]))
        k_fold_train_list.append(pd.DataFrame(xx[k_fold_train_ids_list[ki],:]))
        k_fold_valy_list.append(pd.DataFrame(yy[k_fold_val_ids_list[ki],:]))
        k_fold_trainy_list.append(pd.DataFrame(yy[k_fold_train_ids_list[ki],:]))
    return k_fold_train_list,k_fold_trainy_list,k_fold_val_list,k_fold_valy_list

#### logestic regression class #####

class LogReg():
    def __init__(self,regularization=False):
        self.regularization=regularization
    def p1_given_x(self,x_l,ww,ww0):
        sigma=0
        sigma=np.dot(x_l,ww)
        return 1-(1/(1+np.exp(ww0+sigma)))    
    def fit(self,x,y,x_val=None,y_val=None,itr=20,learning_rate=0.01,reg_par=None):
        self.learning_rate=learning_rate
        self.itr=itr
        self.x=np.array(x)
        self.y=np.array(y)
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val)
        self.feat_size=len(self.x[0])
        self.n_sample=len(self.x)
        self.reg_par=reg_par
        w=np.zeros(self.feat_size)
        self.w=[e+0.01 for e in w]
        self.w0=0.01
        self.train_acc=[]
        self.val_acc=[]
        if self.regularization==True:
            s=[]
            for feat in range(self.feat_size):
                var=np.var(self.x[: , feat])
                s.append(var)
        for itr in range(self.itr):
            a = datetime.datetime.now()    
            print(f'iteration: {itr}' )
            p1=np.zeros(self.n_sample)
            error=np.zeros(self.n_sample)
            sigma_error=np.zeros(self.feat_size)
            sigma_error0=0
            for l in range(self.n_sample):
                p1[l]=self.p1_given_x(self.x[l],self.w,self.w0)
                error[l]=self.y[l]-p1[l]
                for i in range(self.feat_size):
                    sigma_error[i]+=(self.x[l][i]*error[l])
            for l in range(self.n_sample):
                sigma_error0=+(1*error[l])
            self.w0=self.w0+(learning_rate*sigma_error0)
            if self.regularization==True:
                for i in range(self.feat_size):
                    self.w[i]=self.w[i]+(learning_rate*sigma_error[i])-(self.learning_rate*(self.reg_par/(0.01+s[i])*self.w[i]))
            else: 
                for i in range(self.feat_size):
                    self.w[i]=self.w[i]+(learning_rate*sigma_error[i])
            b = datetime.datetime.now()
            c = b - a
            print( int(c.total_seconds()))
            y_pred=np.zeros(len(self.x))
            count=0
            for l in range(len(self.x)):
                y_pred[l]=self.p1_given_x(self.x[l],self.w,self.w0)
                if y_pred[l] >= 0.5 : y_pred[l]=1
                else : y_pred[l]=0
                if y_pred[l]==self.y[l]:
                    count+=1
            print(f' train acc: {count/len(x)}')
        
    def evaluate(self,x_val,y_val):
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val) 
        y_pred=np.zeros(len(self.x_val))
        count=0
        for l in range(len(self.x_val)):
            y_pred[l]=self.p1_given_x(self.x_val[l],self.w,self.w0)
            if y_pred[l] >= 0.5 : y_pred[l]=1
            else : y_pred[l]=0
            if y_pred[l]==self.y_val[l]:
                count+=1
        print(f' acc: {count/len(x_val)}')
        return count/len(x_val)
            

#### 
            
log=LogReg(regularization=False)
log.fit(x_train,y_train,itr=10,reg_par=1) 
log.evaluate(test_data,test_lable) 



































            
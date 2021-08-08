# %load r2_kg_cnn_Hlayer_parse.py
import time,os,csv
import torch.nn as nn
import torchvision
from torchvision.datasets import mnist
import torch,random,sys
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as Data
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.utils import shuffle as shuffle_ds

def rbf(sigma=1):
    def rbf_kernel(x1,x2,sigma):
        X12norm = torch.sum(x1**2,1,keepdims=True)-2*x1@x2.T+torch.sum(x2**2,1,keepdims=True).T
        return torch.exp(-X12norm/(2*sigma**2))
    return lambda x1,x2: rbf_kernel(x1,x2,sigma)

def poly(n=3):
    return lambda x1,x2: (x1 @ x2.T)**n

def grpf(sigma, d):
    return lambda x1,x2: ((d + 2*rbf(sigma)(x1,x2))/(2 + d))**(d+1)

class svm_model_torch:
    def __init__(self, m, n_class, device="cpu"):
        self.device = device
        self.n_svm = n_class * (n_class - 1)//2
        self.m = m # number of samples
        self.n_class = n_class
        self.blacklist = [set() for i in range(self.n_svm)]

        # multiplier
        self.a = torch.zeros((self.n_svm,self.m), device=self.device) # SMO works only when a is initialized to 0
        # bias
        self.b = torch.zeros((self.n_svm,1), device=self.device)

        # kernel function  should input x [n,d] y [m,d] output [n,m]
        # Example of poly kernel: lambda x,y:  torch.matmul(x,y.T)**2
        self.kernel = lambda x,y:  torch.matmul(x,y.T)


        # Binary setting for every SVM,
        # Mij says the SVMj should give
        # Mij label to sample with class i
        self.lookup_matrix=torch.zeros((self.n_class, self.n_svm), device=self.device)

        # The two classes SVMi concerns,
        # lookup_class[i]=[pos, neg]
        self.lookup_class=torch.zeros((self.n_svm, 2), device=self.device)

        k=0
        for i in range(n_class-1):
            for j in range(i+1,n_class):
                self.lookup_class[k, 0]=i
                self.lookup_class[k, 1]=j
                k += 1

        for i in range(n_class):
            for j in range(self.n_svm):
                if i == self.lookup_class[j,0] or i == self.lookup_class[j,1]:
                    if self.lookup_class[j, 0]==i:
                        self.lookup_matrix[i,j]=1.0
                    else:
                        self.lookup_matrix[i,j]=-1.0

    def fit(self, x_np, y_multiclass_np, C, iterations=1, kernel=rbf(1)):
        x_np, y_multiclass_np = shuffle_ds(x_np,y_multiclass_np)
        self.C = C # box constraint
        # use SMO algorithm to fit
        x = torch.from_numpy(x_np).float() if not torch.is_tensor(x_np) else x_np
        x = x.to(self.device)
        self.x = x.to(self.device)

        y_multiclass = torch.from_numpy(y_multiclass_np).view(-1,1) if not torch.is_tensor(y_multiclass_np) else y_multiclass_np
        y_multiclass=y_multiclass.view(-1)
        self.y_matrix = torch.stack([self.cast(y_multiclass, k) for k in range(self.n_svm)],0).to(self.device)
        self.kernel = kernel
        a = self.a
        b = self.b
        for iteration in range(iterations):
#             print("Iteration: ",iteration)
            for k in range(self.n_svm):
                y = self.y_matrix[k, :].view(-1).tolist()
                index = [i for i in range(len(y)) if y[i]!=0]
                shuffle(index)
                traverse = []
                if index is not None:
                    traverse = [i for i in range(0, len(index)-1, 2)]
                    if len(index)>2:
                         traverse += [len(index)-2]
                for i in traverse:
                    if str(index[i])+str(index[i+1]) not in self.blacklist[k]:
                        y1 = y[index[i]]
                        y2 = y[index[i+1]]
                        x1 = x[index[i],:].view(1,-1)
                        x2 = x[index[i+1],:].view(1,-1)
                        a1_old = a[k,index[i]].clone()
                        a2_old = a[k,index[i+1]].clone()

                        if y1 != y2:
                            H = max(min(self.C, (self.C + a2_old-a1_old).item()),0)
                            L = min(max(0, (a2_old-a1_old).item()),self.C)
                        else:
                            H = max(min(self.C, (a2_old + a1_old).item()),0)
                            L = min(max(0, (a2_old + a1_old - self.C).item()),self.C)
                        E1 =  self.g_k(k, x1) - y1
                        E2 =  self.g_k(k, x2) - y2
                        a2_new = torch.clamp(a2_old + y2 * (E1-E2)/self.kernel(x1 - x2,x1 - x2), min=L, max=H)
                        a[k,index[i+1]] = a2_new

                        a1_new = a1_old - y1 * y2 * (a2_new - a2_old)
                        a[k, index[i]] = a1_new

                        b_old = b[k,0]
                        K11 = self.kernel(x1,x1)
                        K12 = self.kernel(x1,x2)
                        K22 = self.kernel(x2,x2)
                        b1_new = b_old - E1 + (a1_old-a1_new)*y1*K11+(a2_old-a2_new)*y2*K12
                        b2_new = b_old - E2 + (a1_old-a1_new)*y1*K12+(a2_old-a2_new)*y2*K22
                        if (0<a1_new) and (a1_new<self.C):
                            b[k,0] = b1_new
                        if (0<a2_new) and (a2_new<self.C):
                            b[k,0] = b2_new
                        if ((a1_new == 0) or (a1_new ==self.C)) and ((a2_new == 0) or (a2_new==self.C)) and (L!=H):
                            b[k,0] = (b1_new + b2_new)/2
                        if b_old == b[k,0] and a[k,index[i]] == a1_old and a[k,index[i+1]] == a2_old:
                            self.blacklist[k].add(str(index[i]) + str(index[i+1]))

    def predict(self,x_np):
        xp = torch.from_numpy(x_np) if not torch.is_tensor(x_np) else x_np
        xp = xp.float().to(self.device)
        k_predicts = (self.y_matrix.to(self.device) * self.a) @ self.kernel(xp,self.x).T  + self.b
        result = torch.argmax(self.lookup_matrix @ k_predicts,axis=0)
        return result.to("cpu").numpy()

    def cast(self, y, k):
        # cast the multiclass label of dataset to
        # the pos/neg (with 0) where pos/neg are what SVMk concerns
        return (y==self.lookup_class[k, 0]).float() - (y==self.lookup_class[k, 1]).float()


    def wTx(self,k,xi):
        # The prediction of SVMk without bias, w^T @ xi
        y = self.y_matrix[k, :].reshape((-1,1))
        a = self.a[k,:].view(-1,1)
        wTx0 =  self.kernel(xi, self.x) @ (y * a)
        return wTx0


    def g_k(self,k,xi):
        # The prediction of SVMk, xi[1,d]
        return self.wTx(k,xi) + self.b[k,0].view(1,1)


    def get_avg_pct_spt_vec(self):
        # the average percentage of support vectors,
        # test error shouldn't be greater than it if traing converge
        return torch.sum((0.0<self.a) & (self.a<self.C)).float().item()/(self.n_svm*self.m)
class SVM_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1) # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(1, 32, (5,5), stride=1, padding=2 # output_size=1+(input_size+2*padding-filter_size)/stride
                          ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                )
        self.conv2=nn.Sequential(nn.Conv2d(32,16,5,stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2))
        self.out=nn.Linear(16*2*8,2)

    def forward(self,input_x,conv2_out_tag = False):
        input_x = self.conv1(input_x)
        input_x = self.conv2(input_x)
        if not conv2_out_tag:
            input_x = input_x.view(input_x.size(0),-1)
            input_x = self.out(input_x)
        return input_x

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
'''obtain data'''
# data=FashionMNIST('../pycharm_workspace/data/')
def wgn(x, snr):
    snr = 10**(snr/10.0)
    
    xpower = np.sum([x0**2 for x0 in x])/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
def train_test_split(rootpath,snr,test_pct=0.3):
    try:
 
        x_data_=[]
        y_data_=[]
        x_data=[]
        y_data=[]
        x_train =[]
        y_train=[]
        x_test=[]
        y_test=[]
        with open(os.path.join(rootpath,'train_data_set.csv'),'r') as f_csv:
            reader = csv.reader(f_csv)
            for row_index,row in enumerate(reader):
                x_data_.append(row)
                y_data_.append(1)
        with open(os.path.join(rootpath,'train_data_set_'+str(snr)+'db.csv'),'r') as f_csv:
            reader = csv.reader(f_csv)
            for row_index,row in enumerate(reader):
                x_data_.append(row)
                y_data_.append(0)
        
        plt.title('Signal')
        ax1 = plt.subplot()
        ax1.plot([i for i in range(int(len(x_data_)/2))],[np.mean(list(map(float,data))) for data in x_data_[:int(len(x_data_)/2)]],'r+', label='Legal signal')
        ax1.plot([i for i in range(int(len(x_data_)/2))],[np.mean(list(map(float,data)))  for data in x_data_[int(len(x_data_)/2):]],'b+', label='Illegal signal')
        ax1.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        while x_data_:
            index_ = random.randint(0,len(x_data_)-1)
            data = x_data_.pop(index_)
            if len(data) == 256:
                x_data.append(np.array(list(map(float,data))))
                y_data.append(y_data_[index_])
        x_train = torch.Tensor(np.reshape(x_data[:20000],(2000,10,16,16)))
        y_train = torch.from_numpy(np.reshape(list(y_data[:20000]),(2000,10)))
        x_test = torch.Tensor(np.reshape(x_data[20000:21000],(100,10,16,16)))
        y_test = torch.from_numpy(np.reshape(list(y_data[20000:21000]),(100,10)))
        return x_train,y_train,x_test,y_test
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Type:'+str(exc_type)+' Reason:'+str(exc_obj)+' Line:'+str(exc_tb.tb_lineno))
def csv_file_recording(filename,data,type):
    '''
    Function csv_file_recording
        => recording data to csv
    :Param filename - csv name
    :Param data - data set after process
    return
    '''
    try:
        csvfile = open(filename, type)
        writer = csv.writer(csvfile)
        #writer.writerow(['TC Category', 'Test Case Name', 'Status1', 'Status2','In 8160 failure list','TC Category2','repeat num'])
        writer.writerows(data)
        csvfile.close()
    except Exception as e:
        print('csv file recoding error: %s' % e)

def cal_accuracy(model,x_test,y_test,samples=10000):
    y_pred=model(x_test[:samples])
#     print(y_pred)
    y_pred_=list(map(lambda x:np.argmax(x),y_pred.data.numpy()))
#     print(y_pred_)
    acc=sum(y_pred_==y_test.numpy()[:samples])/samples
    return acc


def train(snr):
    num_epoch=1000
    x_train,y_train,x_test,y_test=train_test_split('./data/a_save_to_mysql_data',snr,0.2)


    for i in range(10):
        x_input_data = [[data[0].mean()] for data in x_train[i:(i+1)*200].detach().numpy()]
        y_input_data = [[data[0]] for data in y_train[i:(i+1)*200].detach().numpy()]

        x_input_data = np.array(x_input_data)
        y_input_data = np.array(y_input_data)
        m = len(x_input_data)
        c = len(np.unique(y_input_data))
        svm = svm_model_torch(m,c)
        svm.fit(x_input_data,y_input_data,1,100)
    #     svm_test_data = model(torch.unsqueeze(x_test,dim=1))
        input_data = []
        input_data = np.array([[data.mean()] for data in x_train[i:(i+1)*200].detach().numpy()])

        predict_result = svm.predict(input_data)
        index  = 0
        for ci, data in enumerate(predict_result):
            if data == y_train[i:(i+1)*200].detach().numpy()[ci][0]:
                index += 1

    print('SVM accuracy:',index/len(y_train[i:(i+1)*200]))

def auto_train(rootpath):
    for snr in [5,10,15,20,25]:
        train(snr)
if __name__=='__main__':
    # train()
    auto_train('./data/a_save_to_mysql_data')
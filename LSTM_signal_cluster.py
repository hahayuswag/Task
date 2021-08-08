'''
本程序实现用LSTM对Signal进行分类
'''
 
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
# import matplotlib.pyplot as plt
 
 
# Hyper parameter
EPOCH = 1
LR = 0.001    # learning rate
BATCH_SIZE = 10
 
 
 
class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):                  # x's shape (batch_size, 序列长度, 序列中每个数据的长度)
        out, _ = self.lstm(x)              # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]                # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
                                           # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)             # 经过线性层后，out的shape为(batch_size, n_class)
        return out
for snr in [5,10,15,20,25]:

    model = LSTMnet(16, 64, 2, 2)             # 图片大小28*28，lstm的每个隐藏层64个节点，2层隐藏层
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    x_train,y_train,x_test,y_test=train_test_split('/Users/jinwei_yu/Documents/03_r2_related_procedure/06_ML_learning/01_neural_machine_translation/selfcreate/data/a_save_to_mysql_data',snr,0.2)
    # training and testing
    for epoch in range(EPOCH):
        for iteration in range(len(x_train)):
            train_x = x_train[iteration]
            train_y = y_train[iteration]
            train_x = train_x.squeeze()        # after squeeze, train_x's shape (BATCH_SIZE,28,28),
                                               # 第一个28是序列长度，第二个28是序列中每个数据的长度。
            output = model(train_x)
            loss = criterion(output, train_y)  # cross entropy loss
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if iteration % 100 == 0:
                test_output = model(x_test[0])
                predict_y = torch.max(test_output, 1)[1].numpy()
                accuracy = float((predict_y == y_test[0].numpy()).astype(int).sum()) / float(y_test[0].size(0))
                print('epoch:{:<2d} | iteration:{:<4d} | loss:{:<6.4f} | accuracy:{:<4.2f}'.format(epoch, iteration, loss, accuracy))


    # print 10 predictions from test data
    Accury = 0
    count = 0
    for x_test_ in x_test:
        test_out = model(x_test_)
        pred_y = torch.max(test_out, dim=1)[1].data.numpy()
        for index,v in enumerate(pred_y):
            if v == y_test[1].numpy()[index]:
                Accury += 1
            count += 1
    #     print('The predict number is:')
    #     print(pred_y)
    #     print('The real number is:')
    #     print(y_test[1].numpy())
    print(Accury/count)
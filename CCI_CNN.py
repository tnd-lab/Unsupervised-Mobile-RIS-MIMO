#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
import time
import math

current_time_seconds = time.time()
current_time = time.ctime(current_time_seconds)
print(f"Current time: {current_time}")

device = "cpu"

B = 3 # number of phase-shift quantization bits
M = 4 # number of Source antennas
N = 12 # number of RIS elements
U = 3 # number of users
kappa = 1
quant = 2 * math.pi / (2 ** B)

PowSrc = 10 ** ((20-30)/10)
print(PowSrc)
sigma2 = 10 ** ((-80-30)/10)
print(sigma2)
BW = th.tensor(50 * (10**6), dtype=th.double).to(device)
print('bandwidth = ',BW.item())
wo1 = 10 ** ((-85-30)/10)
print(wo1)
wo2 = 10 ** ((-90-30)/10)
print(wo2)
wo3 = 10 ** ((-95-30)/10)
print(wo3)

epoch_size = 100
learning_rate = 0.001
wd = 1e-6
n_hidden = 64
ksz = 3
n_out = N+U+2*M*U

train_bsz = 4
n_row_in = U
n_col_in = 2*N

class net_multi_task(nn.Module):
    def __init__(self):
        super(net_multi_task, self).__init__()

        self.lay1conv1 = nn.Conv2d(1, n_hidden, ksz, stride=1, padding=1, dtype=th.double).to(device)
        self.lay2pool1 = nn.AvgPool2d(ksz, stride=1, padding=1).to(device)
        self.lay3conv2 = nn.Conv2d(n_hidden, n_hidden, ksz, stride=1, padding=1, dtype=th.double).to(device)
        self.lay4pool2 = nn.AvgPool2d(ksz, stride=1, padding=1).to(device)
        self.lay5flat = nn.Flatten().to(device)
        self.lay6lin = nn.Linear(n_hidden*n_row_in*n_col_in, n_hidden, dtype=th.double).to(device)
        self.lay7out = nn.Linear(n_hidden, n_out, dtype=th.double).to(device)
        self.lay8act = nn.Sigmoid().to(device)

    def forward(self, x):
        out_lay1_conv1 = self.lay1conv1(x)
        out_lay2_pool1 = self.lay2pool1(out_lay1_conv1)
        out_lay3_conv2 = self.lay3conv2(out_lay2_pool1)
        out_lay4_pool2 = self.lay4pool2(out_lay3_conv2)
        out_lay5_flat = self.lay5flat(out_lay4_pool2)
        out_lay6_lin = self.lay6lin(out_lay5_flat)
        out_lay7_out = self.lay7out(out_lay6_lin)
        out_lay8_act = self.lay8act(out_lay7_out)
        return out_lay8_act

HSR_csv = pd.read_csv('channel\HSR_S3_R8_M4_N12.csv')
HSR_data = th.tensor(HSR_csv.values, dtype=th.double)
print('The shape of HSR data is: ', HSR_data.shape)
HSR_real = HSR_data[0:N,0:M] * th.cos(HSR_data[0:N,M:2*M])
HSR_imag = HSR_data[0:N,0:M] * th.sin(HSR_data[0:N,M:2*M])
HSR = th.complex(HSR_real, HSR_imag).to(device)

HRD_csv_train = pd.read_csv('channel\HRD_GMM_S3_R8_M4_N12_U3_new.csv')
HRD_data_train = th.tensor(HRD_csv_train.values, dtype=th.double)
print('The shape of HRD data is: ', HRD_data_train.shape)
data_len = int(len(HRD_data_train)/U)
print('The length of HRD data is: ', data_len)

HRD_real = HRD_data_train[:,0:N] * th.cos(HRD_data_train[:,N:2*N])
HRD_imag = HRD_data_train[:,0:N] * th.sin(HRD_data_train[:,N:2*N])
HRD = th.complex(HRD_real, HRD_imag).to(device)
print('The shape of HRD is: ', HRD.shape)
num_batch = int(data_len/train_bsz)
print('The number of HRD data batch is: ', num_batch)
HRD_batch = HRD.reshape(data_len,U,N)
print('The shape of HRD train is: ', HRD_batch.shape)
HRD_train = HRD_batch.reshape(num_batch,train_bsz,U,N)
print('The shape of HRD train is: ', HRD_train.shape)

HRD_data_train[:,0:N] = HRD_data_train[:,0:N] * (10**5)
HRD_data_train[:,N:2*N] = (HRD_data_train[:,N:2*N] + math.pi)
dataset = HRD_data_train.reshape(data_len,1,n_row_in,n_col_in)
print('The shape of the data set is: ', dataset.shape)
train_dataloader = th.utils.data.DataLoader(dataset, batch_size=train_bsz, shuffle=False, num_workers=0)


# # Unsupervised Learning

# In[79]:


test_bsz = 1
HRD_data_train = th.tensor(HRD_csv_train.values, dtype=th.double)
data_len = int(len(HRD_data_train)/U)
HRD_real = HRD_data_train[:,0:N] * th.cos(HRD_data_train[:,N:2*N])
HRD_imag = HRD_data_train[:,0:N] * th.sin(HRD_data_train[:,N:2*N])
HRD = th.complex(HRD_real, HRD_imag).to(device)
num_batch = int(data_len/test_bsz)
HRD_batch = HRD.reshape(data_len,U,N)
HRD_test = HRD_batch.reshape(num_batch,test_bsz,U,N)
print('The shape of HRD test is: ', HRD_test.shape)
HRD_data_train[:,0:N] = HRD_data_train[:,0:N] * (10**5)
HRD_data_train[:,N:2*N] = (HRD_data_train[:,N:2*N] + math.pi)
dataset = HRD_data_train[0:3000,:].reshape(1000,1,n_row_in,n_col_in)
print('The shape of the data set is: ', dataset.shape)
test_dataloader = th.utils.data.DataLoader(dataset, batch_size=test_bsz, shuffle=False, num_workers=0)

init_bsz = 1
out_csv = pd.read_csv('channel\OUT_GMM_S3_R8_M4_N12_U3_new.csv')
out_data = th.tensor(out_csv.values, dtype=th.double)
print('The shape of output data is: ', out_data.shape)
data_len = int(len(out_data)/1)
print('The length of output data is: ', data_len)
supdataset = th.utils.data.TensorDataset(dataset, out_data)
suploader = th.utils.data.DataLoader(supdataset, batch_size=init_bsz, shuffle=False)


# In[80]:


num_pred = 10
binit = 20
totalSR = 0
tot_time = 0
out_data = np.zeros([num_pred, epoch_size+4+binit])
criterion = nn.MSELoss()

for c in range(num_pred):
    time_start = time.perf_counter()
    source_model = net_multi_task()
    source_optim = th.optim.Adam(source_model.parameters(), lr=learning_rate, weight_decay=wd)
    epoch_loss = []
    
    for epoch in range(binit):
        total_loss = []
        b = 0
        for xb, yb in suploader:
            source_optim.zero_grad()
            pred = source_model(xb)
            diff = criterion(pred, yb)
            diff.backward()
            source_optim.step()
            init_ris_phs = pred[:,0:N].cpu().detach().numpy()
            init_ris=(kappa*th.cos(2*math.pi*pred[:,0:N]))+(1j*kappa*th.sin(2*math.pi*pred[:,0:N]))
            init_pow = pred[:,N:N+U] * PowSrc / th.sum(pred[:,N:N+U])
 
            init_bf1_real = pred[:,N+U:N+U+M] * th.cos(2*math.pi*pred[:,N+U+M*U:N+U+M*U+M])
            init_bf1_imag = pred[:,N+U:N+U+M] * th.sin(2*math.pi*pred[:,N+U+M*U:N+U+M*U+M])
            init_bf1 = th.complex(init_bf1_real, init_bf1_imag)
            init_norm1 = init_bf1 / th.norm(init_bf1)
            init_bf2_real = pred[:,N+U+M:N+U+2*M] * th.cos(2*math.pi*pred[:,N+U+M*U+M:N+U+M*U+2*M])
            init_bf2_imag = pred[:,N+U+M:N+U+2*M] * th.sin(2*math.pi*pred[:,N+U+M*U+M:N+U+M*U+2*M])
            init_bf2 = th.complex(init_bf2_real, init_bf2_imag)
            init_norm2 = init_bf2 / th.norm(init_bf2)
            init_bf3_real = pred[:,N+U+2*M:N+U+3*M] * th.cos(2*math.pi*pred[:,N+U+M*U+2*M:N+U+M*U+3*M])
            init_bf3_imag = pred[:,N+U+2*M:N+U+3*M] * th.sin(2*math.pi*pred[:,N+U+M*U+2*M:N+U+M*U+3*M])
            init_bf3 = th.complex(init_bf3_real, init_bf3_imag)
            init_norm3 = init_bf3 / th.norm(init_bf3)
             
            init_mm1a = th.sum(th.mm(init_ris*HRD_test[b,:,0,:],HSR)*init_norm1)
            init_sr1a = init_pow[:,0] * (init_mm1a.real ** 2 + init_mm1a.imag ** 2)
            init_mm1b = th.sum(th.mm(init_ris*HRD_test[b,:,0,:],HSR)*init_norm2)
            init_sr1b = init_pow[:,1] * (init_mm1b.real ** 2 + init_mm1b.imag ** 2)
            init_mm1c = th.sum(th.mm(init_ris*HRD_test[b,:,0,:],HSR)*init_norm3)
            init_sr1c = init_pow[:,2] * (init_mm1c.real ** 2 + init_mm1c.imag ** 2)
            init_SNR1 = init_sr1a / ((init_sr1b*(1+wo1)) + (init_sr1c*(1+wo1)) + (init_sr1a*wo1) + sigma2)
            init_mm2a = th.sum(th.mm(init_ris*HRD_test[b,:,1,:],HSR)*init_norm2)
            init_sr2a = init_pow[:,1] * (init_mm2a.real ** 2 + init_mm2a.imag ** 2)
            init_mm2b = th.sum(th.mm(init_ris*HRD_test[b,:,1,:],HSR)*init_norm1)
            init_sr2b = init_pow[:,0] * (init_mm2b.real ** 2 + init_mm2b.imag ** 2)
            init_mm2c = th.sum(th.mm(init_ris*HRD_test[b,:,1,:],HSR)*init_norm3)
            init_sr2c = init_pow[:,2] * (init_mm2c.real ** 2 + init_mm2c.imag ** 2)
            init_SNR2 = init_sr2a / ((init_sr2b*(1+wo2)) + (init_sr2c*(1+wo2)) + (init_sr2a*wo2) + sigma2)
            init_mm3a = th.sum(th.mm(init_ris*HRD_test[b,:,2,:],HSR)*init_norm3)
            init_sr3a = init_pow[:,2] * (init_mm3a.real ** 2 + init_mm3a.imag ** 2)
            init_mm3b = th.sum(th.mm(init_ris*HRD_test[b,:,2,:],HSR)*init_norm1)
            init_sr3b = init_pow[:,0] * (init_mm3b.real ** 2 + init_mm3b.imag ** 2)
            init_mm3c = th.sum(th.mm(init_ris*HRD_test[b,:,2,:],HSR)*init_norm2)
            init_sr3c = init_pow[:,1] * (init_mm3c.real ** 2 + init_mm3c.imag ** 2)
            init_SNR3 = init_sr3a / ((init_sr3b*(1+wo3)) + (init_sr3c*(1+wo3)) + (init_sr3a*wo3) + sigma2)
            init_loss = -1*BW*(th.log2(1+init_SNR1)+th.log2(1+init_SNR2)+th.log2(1+init_SNR3))
            avg_loss = sum(init_loss) / init_bsz
            total_loss.append(avg_loss.item())

        epoch_loss.append(sum(total_loss)/len(total_loss))
        
    
    for a in range(epoch_size):
        total_loss = []
        for b, train_inp in enumerate(train_dataloader):
            batch = train_inp.to(device)
            source_optim.zero_grad()
            tr_out = source_model(batch)
            train_ris_phs = tr_out[:,0:N].cpu().detach().numpy()
            train_ris=(kappa*th.cos(2*math.pi*tr_out[:,0:N]))+(1j*kappa*th.sin(2*math.pi*tr_out[:,0:N]))
            train_pow = tr_out[:,N:N+U] * PowSrc / th.sum(tr_out[:,N:N+U])
 
            train_bf1_real = tr_out[:,N+U:N+U+M] * th.cos(2*math.pi*tr_out[:,N+U+M*U:N+U+M*U+M])
            train_bf1_imag = tr_out[:,N+U:N+U+M] * th.sin(2*math.pi*tr_out[:,N+U+M*U:N+U+M*U+M])
            train_bf1 = th.complex(train_bf1_real, train_bf1_imag)
            train_norm1 = train_bf1 / th.norm(train_bf1)
            train_bf2_real = tr_out[:,N+U+M:N+U+2*M] * th.cos(2*math.pi*tr_out[:,N+U+M*U+M:N+U+M*U+2*M])
            train_bf2_imag = tr_out[:,N+U+M:N+U+2*M] * th.sin(2*math.pi*tr_out[:,N+U+M*U+M:N+U+M*U+2*M])
            train_bf2 = th.complex(train_bf2_real, train_bf2_imag)
            train_norm2 = train_bf2 / th.norm(train_bf2)
            train_bf3_real = tr_out[:,N+U+2*M:N+U+3*M] * th.cos(2*math.pi*tr_out[:,N+U+M*U+2*M:N+U+M*U+3*M])
            train_bf3_imag = tr_out[:,N+U+2*M:N+U+3*M] * th.sin(2*math.pi*tr_out[:,N+U+M*U+2*M:N+U+M*U+3*M])
            train_bf3 = th.complex(train_bf3_real, train_bf3_imag)
            train_norm3 = train_bf3 / th.norm(train_bf3)
             
            train_mm1a = th.sum(th.mm(train_ris*HRD_train[b,:,0,:],HSR)*train_norm1)
            train_sr1a = train_pow[:,0] * (train_mm1a.real ** 2 + train_mm1a.imag ** 2)
            train_mm1b = th.sum(th.mm(train_ris*HRD_train[b,:,0,:],HSR)*train_norm2)
            train_sr1b = train_pow[:,1] * (train_mm1b.real ** 2 + train_mm1b.imag ** 2)
            train_mm1c = th.sum(th.mm(train_ris*HRD_train[b,:,0,:],HSR)*train_norm3)
            train_sr1c = train_pow[:,2] * (train_mm1c.real ** 2 + train_mm1c.imag ** 2)
            train_SNR1 = train_sr1a / ((train_sr1b*(1+wo1)) + (train_sr1c*(1+wo1)) + (train_sr1a*wo1) + sigma2)
            train_mm2a = th.sum(th.mm(train_ris*HRD_train[b,:,1,:],HSR)*train_norm2)
            train_sr2a = train_pow[:,1] * (train_mm2a.real ** 2 + train_mm2a.imag ** 2)
            train_mm2b = th.sum(th.mm(train_ris*HRD_train[b,:,1,:],HSR)*train_norm1)
            train_sr2b = train_pow[:,0] * (train_mm2b.real ** 2 + train_mm2b.imag ** 2)
            train_mm2c = th.sum(th.mm(train_ris*HRD_train[b,:,1,:],HSR)*train_norm3)
            train_sr2c = train_pow[:,2] * (train_mm2c.real ** 2 + train_mm2c.imag ** 2)
            train_SNR2 = train_sr2a / ((train_sr2b*(1+wo2)) + (train_sr2c*(1+wo2)) + (train_sr2a*wo2) + sigma2)
            train_mm3a = th.sum(th.mm(train_ris*HRD_train[b,:,2,:],HSR)*train_norm3)
            train_sr3a = train_pow[:,2] * (train_mm3a.real ** 2 + train_mm3a.imag ** 2)
            train_mm3b = th.sum(th.mm(train_ris*HRD_train[b,:,2,:],HSR)*train_norm1)
            train_sr3b = train_pow[:,0] * (train_mm3b.real ** 2 + train_mm3b.imag ** 2)
            train_mm3c = th.sum(th.mm(train_ris*HRD_train[b,:,2,:],HSR)*train_norm2)
            train_sr3c = train_pow[:,1] * (train_mm3c.real ** 2 + train_mm3c.imag ** 2)
            train_SNR3 = train_sr3a / ((train_sr3b*(1+wo3)) + (train_sr3c*(1+wo3)) + (train_sr3a*wo3) + sigma2)
            train_loss = -1*BW*(th.log2(1+train_SNR1)+th.log2(1+train_SNR2)+th.log2(1+train_SNR3))
            avg_loss = sum(train_loss) / train_bsz
            avg_loss.backward()
            source_optim.step()
            total_loss.append(avg_loss.item())

        epoch_loss.append(sum(total_loss)/len(total_loss))
        
    train_time = time.perf_counter() - time_start
    time_start = time.perf_counter()
    with th.no_grad():
        predSR = 0
        for b, test_inp in enumerate(test_dataloader):
#         for test_inp in test_dataloader:
            batch = test_inp.to(device)
            source_model.eval()
            out = source_model(batch)
            out_digit = quant * th.round((2*math.pi*out[:,0:N])/quant)
            out_phase = out.cpu().detach().numpy()
            test_ris = (kappa * th.cos(out_digit)) + (1j * kappa * th.sin(out_digit))
            test_pow = out[:,N:N+U] * PowSrc / th.sum(out[:,N:N+U])

            test_bf1_real = out[:,N+U:N+U+M] * th.cos(2*math.pi*out[:,N+U+M*U:N+U+M*U+M])
            test_bf1_imag = out[:,N+U:N+U+M] * th.sin(2*math.pi*out[:,N+U+M*U:N+U+M*U+M])
            test_bf1 = th.complex(test_bf1_real, test_bf1_imag)
            test_norm1 = test_bf1 / th.norm(test_bf1)
            test_bf2_real = out[:,N+U+M:N+U+2*M] * th.cos(2*math.pi*out[:,N+U+M*U+M:N+U+M*U+2*M])
            test_bf2_imag = out[:,N+U+M:N+U+2*M] * th.sin(2*math.pi*out[:,N+U+M*U+M:N+U+M*U+2*M])
            test_bf2 = th.complex(test_bf2_real, test_bf2_imag)
            test_norm2 = test_bf2 / th.norm(test_bf2)
            test_bf3_real = out[:,N+U+2*M:N+U+3*M] * th.cos(2*math.pi*out[:,N+U+M*U+2*M:N+U+M*U+3*M])
            test_bf3_imag = out[:,N+U+2*M:N+U+3*M] * th.sin(2*math.pi*out[:,N+U+M*U+2*M:N+U+M*U+3*M])
            test_bf3 = th.complex(test_bf3_real, test_bf3_imag)
            test_norm3 = test_bf3 / th.norm(test_bf3)
           
            test_mm1a = th.sum(th.mm(test_ris*HRD_test[b,:,0,:], HSR)*test_norm1)
            test_sr1a = test_pow[:,0] * (test_mm1a.real ** 2 + test_mm1a.imag ** 2)
            test_mm1b = th.sum(th.mm(test_ris*HRD_test[b,:,0,:], HSR)*test_norm2)
            test_sr1b = test_pow[:,1] * (test_mm1b.real ** 2 + test_mm1b.imag ** 2)
            test_mm1c = th.sum(th.mm(test_ris*HRD_test[b,:,0,:], HSR)*test_norm3)
            test_sr1c = test_pow[:,2] * (test_mm1c.real ** 2 + test_mm1c.imag ** 2)
            test_SNR1 = test_sr1a / ((test_sr1b*(1+wo1)) + (test_sr1c*(1+wo1)) + (test_sr1a*wo1) + sigma2)
            test_mm2a = th.sum(th.mm(test_ris*HRD_test[b,:,1,:], HSR)*test_norm2)
            test_sr2a = test_pow[:,1] * (test_mm2a.real ** 2 + test_mm2a.imag ** 2)
            test_mm2b = th.sum(th.mm(test_ris*HRD_test[b,:,1,:], HSR)*test_norm1)
            test_sr2b = test_pow[:,0] * (test_mm2b.real ** 2 + test_mm2b.imag ** 2)
            test_mm2c = th.sum(th.mm(test_ris*HRD_test[b,:,1,:], HSR)*test_norm3)
            test_sr2c = test_pow[:,2] * (test_mm2c.real ** 2 + test_mm2c.imag ** 2)
            test_SNR2 = test_sr2a / ((test_sr2b*(1+wo2)) + (test_sr2c*(1+wo2)) + (test_sr2a*wo2) + sigma2)
            test_mm3a = th.sum(th.mm(test_ris*HRD_test[b,:,2,:], HSR)*test_norm3)
            test_sr3a = test_pow[:,2] * (test_mm3a.real ** 2 + test_mm3a.imag ** 2)
            test_mm3b = th.sum(th.mm(test_ris*HRD_test[b,:,2,:], HSR)*test_norm1)
            test_sr3b = test_pow[:,0] * (test_mm3b.real ** 2 + test_mm3b.imag ** 2)
            test_mm3c = th.sum(th.mm(test_ris*HRD_test[b,:,2,:], HSR)*test_norm2)
            test_sr3c = test_pow[:,1] * (test_mm3c.real ** 2 + test_mm3c.imag ** 2)
            test_SNR3 = test_sr3a / ((test_sr3b*(1+wo3)) + (test_sr3c*(1+wo3)) + (test_sr3a*wo3) + sigma2)
            SR = BW*(th.log2(1+test_SNR1) + th.log2(1+test_SNR2) + th.log2(1+test_SNR3))
            avgSR = sum(SR) / test_bsz
            predSR = predSR + avgSR

    predSR = predSR / data_len
    totalSR = totalSR + predSR
    elapsed_time = time.perf_counter() - time_start
    tot_time = tot_time + elapsed_time
#     print('Elapsed %.4f seconds' % elapsed_time)
    print(f'{c} Sum Rate {predSR} Train {train_time} Elapsed {elapsed_time} seconds')

    out_data [c,0] = c
    out_data [c,1] = predSR.item()
    out_data [c,2] = train_time
    out_data [c,3] = elapsed_time
    out_data [c,4:epoch_size+4+binit] = epoch_loss

np.savetxt("out_GMM_S3_R8_M4_N12_U3_251020_20_4_L2.csv", out_data, delimiter=",")


# In[81]:


plt.plot(epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

average_total_SR = totalSR / num_pred
average_tot_time = tot_time / num_pred
print(f'Sum Rate {average_total_SR.item()} Elapsed {average_tot_time} seconds')

current_time_seconds = time.time()
current_time = time.ctime(current_time_seconds)
print(f"Current time: {current_time}")


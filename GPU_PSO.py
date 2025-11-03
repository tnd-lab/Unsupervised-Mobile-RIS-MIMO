#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch, math, numpy as np
import pandas as pd
import time

current_time_seconds = time.time()
current_time = time.ctime(current_time_seconds)
print(f"Current time: {current_time}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# fdtype = torch.float32
# cdtype = torch.complex64

N, M, U, B = 4, 4, 3, 3
kap = 1
Q = 2**B
PSpool = torch.linspace(0, 2*math.pi*(1 - 1/Q), Q, device=device, dtype=torch.double)
e_jP = (kap*torch.cos(PSpool))+(1j*kap*torch.sin(PSpool))

numdata = 1000
numRep = 10
numPar = 10 * M * N * U
numItr = 5
w = 0.9
c1 = 1.2
c2 = 1.2

HSR_csv = pd.read_csv('channel\HSR_S3_R8_M4_N4.csv')
inpHSR_np = torch.tensor(HSR_csv.values, dtype=torch.double)
mag_HSR = torch.tensor(inpHSR_np[:, :M], device=device, dtype=torch.double)
phs_HSR = torch.tensor(inpHSR_np[:, M:2*M], device=device, dtype=torch.double)
HSR_real = mag_HSR * torch.cos(phs_HSR)
HSR_imag = mag_HSR * torch.sin(phs_HSR)
HSR = torch.complex(HSR_real,HSR_imag)
# print("HSR", HSR.shape, HSR)

HRD_csv = pd.read_csv('channel\HRD_RPG_S3_R8_M4_N4_U3_new.csv')
inpHRD_np = torch.tensor(HRD_csv.values, dtype=torch.double)
mag_HRD = torch.tensor(inpHRD_np[:, :N], device=device, dtype=torch.double)
phs_HRD = torch.tensor(inpHRD_np[:, N:2*N], device=device, dtype=torch.double)
HRD_real = mag_HRD * torch.cos(phs_HRD)
HRD_imag = mag_HRD * torch.sin(phs_HRD)
allHRD = torch.complex(HRD_real,HRD_imag)
print("allHRD", allHRD.shape)

PowSrc = 10**((20-30)/10)   # db2pow(20-30)
BW     = 50e6
sigma2 = 10**((-80-30)/10)  # db2pow(-110)

SRcc = torch.zeros(numdata, dtype=torch.double, device=device)
SRidx = torch.zeros(numdata, dtype=torch.double, device=device)
SRdiff = torch.zeros(numdata, dtype=torch.double, device=device)
timePSO = torch.zeros(numdata, dtype=torch.double, device=device)
out_data = np.zeros([numRep, 9])

for rep in range(numRep):

    for idx in range(numdata):
        hRD = allHRD[U*(idx):U*(idx+1), :].T.contiguous()                   # (N, U) complex
        time_start = time.perf_counter()

        Cn = kap * hRD.sum(dim=1) * HSR.sum(dim=1)
        scores = (Cn.unsqueeze(1) * e_jP.unsqueeze(0)).real
        best_idx = scores.argmax(dim=1)
        PSmy = PSpool[best_idx]
        PSmy = torch.remainder(PSmy, 2*math.pi)

        expPS = (kap*torch.cos(PSmy))+(1j*kap*torch.sin(PSmy))
        A = (expPS[:, None] * HSR)
        Hmat = (hRD.transpose(0, 1) @ A) * kap
        norms = torch.linalg.norm(Hmat, dim=1, keepdim=True) + 1e-12
        BFmy = torch.conj(Hmat) / norms
        BFcc = BFmy.transpose(0, 1).contiguous()
        BF_flat = BFmy.reshape(-1, 1)

        s = (Hmat * BFcc.transpose(0, 1)).sum(dim=1)
        PA = s.real ** 2 + s.imag ** 2
        sumPA = torch.sum(PA)
        PAmy = (PowSrc / sumPA) * PA

        S = Hmat @ BFcc
        P = S.real ** 2 + S.imag ** 2
        num = PAmy * torch.diag(P)
        interf = (P * PAmy.unsqueeze(0)).sum(dim=1) - num
        SNR = num / (interf + sigma2)
        SRmy = BW * torch.log2(1 + SNR).sum()
        SRcc[idx] = SRmy
        train_time = time.perf_counter() - time_start

        SRvec = torch.full((numPar,), SRmy, dtype=torch.double, device=device)
        absBF = (torch.sqrt(BF_flat.real ** 2 + BF_flat.imag ** 2)).squeeze()
        angleBF = (torch.atan(BF_flat.imag / BF_flat.real)).squeeze()
        base_vec = torch.zeros(N+U+(2*M*U), dtype=torch.double, device=device)
        base_vec [0:N] = PSmy
        base_vec [N:N+U] = PAmy
        base_vec [N+U : N+U+(M*U)] = absBF
        base_vec [N+U+(M*U):N+U+(2*M*U)] = angleBF
        # Broadcast base_vec to all particles (numPar columns)
        Pos = base_vec.unsqueeze(1).expand(-1, numPar).clone()
        SRglo = SRmy
        PTglo = base_vec
        SRpar = SRvec
        PTpar = Pos
        SRpso = SRglo
        PTpso = PTglo
        Vel = torch.rand_like(Pos)

        time_start = time.perf_counter()
        for itr in range(numItr):

            r1 = torch.rand_like(Pos)
            r2 = torch.rand_like(Pos)
            Vel = w*Vel + c1*r1*(PTpar-Pos) + c2*r2*(PTglo.unsqueeze(1) - Pos)
            Pos = Pos + Vel

            PA_PSO = PAmy.unsqueeze(1).expand(-1, numPar).clone()
            BFmag  = Pos[N+U:N+U+M*U, :].view(M, U, -1)                  # (M,U,P)
            BFphs  = torch.remainder(2*math.pi * Pos[N+U+M*U:N+U+2*M*U, :], 2*math.pi).view(M, U, -1)
            BFcomp = (BFmag * torch.cos(BFphs)) + (1j*BFmag * torch.sin(BFphs))
            BF_PSO = BFcomp / (BFcomp.norm(dim=0, keepdim=True) + 1e-12) # (M,U,P)

            PSq = expPS.unsqueeze(1).expand(-1, numPar).clone()
            A_PSO = PSq.unsqueeze(1) * HSR.unsqueeze(-1)                 # (N,M,P)
            Hmat = torch.einsum('nu,nmp->ump', hRD.conj(), A_PSO) * kap  # (U,M,P)

            S_PSO = torch.einsum('ump,mtp->utp', Hmat, BF_PSO)           # (U,U,P)
            Ppow = S_PSO.real ** 2 + S_PSO.imag ** 2

            diagP = torch.diagonal(Ppow, dim1=0, dim2=1)                 # (U,P)
            num   = PA_PSO * torch.transpose(diagP,0,1)
            interf = (Ppow * PA_PSO.unsqueeze(0)).sum(dim=1) - num                      # (U,P)
            SINR   = num / (interf + sigma2)
            SRvec = BW * torch.log2(1 + SINR).sum(dim=0)

            improved = SRvec > SRpar
            SRpar = torch.maximum(SRpar, SRvec)
            if improved.any():
                PTpar[:, improved] = Pos[:, improved]

            max_now, idx_now = torch.max(SRvec, dim=0)
            if max_now > SRglo:
                SRglo = max_now
                PTglo = Pos[:, idx_now].clone()

            if SRglo > SRpso:
                SRpso = SRglo
                PTpso = PTglo.clone()

        train_time = time.perf_counter() - time_start
        timePSO[idx] = train_time
        SRidx[idx] = SRpso
        SRdiff[idx] = SRpso - SRmy

    SRinit = torch.sum(SRcc) / numdata
    SRave = torch.sum(SRidx) / numdata
    print(rep, "SRave", SRave)
    count = (SRdiff > 0).sum()
    SRtot = torch.sum(SRdiff)
    SRfix = (torch.sum(SRdiff) / count) + SRinit
    print(SRfix)
    time_ave = torch.sum(timePSO) / numdata
    time_max = torch.max(timePSO)
    time_min = torch.min(timePSO)

    out_data [rep,0] = rep
    out_data [rep,1] = SRinit.item()
    out_data [rep,2] = SRave.item()
    out_data [rep,3] = SRfix.item()
    out_data [rep,4] = SRtot.item()
    out_data [rep,5] = count.item()
    out_data [rep,6] = time_max.item()
    out_data [rep,7] = time_ave.item()
    out_data [rep,8] = time_min.item()

np.savetxt("out_RPG_S3_R8_M4_N4_U3_251027_cpu_10.csv", out_data, delimiter=",")

current_time_seconds = time.time()
current_time = time.ctime(current_time_seconds)
print(f"Current time: {current_time}")


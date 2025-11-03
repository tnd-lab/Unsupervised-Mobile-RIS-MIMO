close all
clear
clc

fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
fprintf('\n')

PowSrc = db2pow(20-30);
BW = 50 * (10^6);
sigma2 = db2pow(-80-30);

M = 4;
N = 16;
U = 3;
B = 3;
kap = 1;
numdata = 1000;

load HSR_S3_R8_M4_N16.csv
inpHSR = HSR_S3_R8_M4_N16(2:N+1,:);
HSR = inpHSR(:,1:M).*exp(1i*inpHSR(:,M+1:M*2));

load HRD_RPG_S3_R8_M4_N16_U3_new.csv
inpHRD = HRD_RPG_S3_R8_M4_N16_U3_new(2:numdata*U+1,:);
allHRD = inpHRD(:,1:N).*exp(1i*inpHRD(:,N+1:N*2));

PSpool = wrapTo2Pi((1:2^B)*2*pi/(2^B));
numrep = 10;

wo = zeros([U,1]);
for ue = 1:U
    wo(ue) = db2pow(-80-(5*ue)-30);
end

SRmy3 = zeros([numdata,numrep]);
time_my3 = zeros([numrep,numdata]);
timeSR3 = zeros([numrep,numdata]);
Pos = zeros([numdata,N+U+2*M*U]);
BF_flat = zeros([numdata,M*U]);

for idx = 1:numdata
hRD = transpose(allHRD(U*(idx-1)+1:U*idx,:));

for rep = 1:numrep

PSmy = zeros([N,1]);
tic
for nn = 1:N
    MaxSum = 0;
    for qq = 1:2^B
        Mat = real(transpose(hRD(nn,:))*kap*exp(1i*PSpool(qq))*HSR(nn,:));
        MatSum = sum(sum(Mat));
        if MaxSum <= MatSum
            MaxSum = MatSum;
            PSmy(nn) = PSpool(qq);
        end
    end
end

BF = zeros([U,M]);
for ue = 1:U
    hRDrisHSR = transpose(hRD(:,ue))*kap*diag(exp(1i*PSmy))*HSR;
    BF(ue,:) = conj(hRDrisHSR)/norm(hRDrisHSR);
    BF_flat(idx,M*(ue-1)+1:M*ue) = BF(ue,:);
end

PAmy3 = zeros([U,1]);
for ue = 1:U
    PAmy3(ue) = (1+wo(ue))*abs(transpose(hRD(:,ue))*kap*diag(exp(1i*PSmy))*HSR*transpose(BF(ue,:)))^2;
end
PAmy3 = PAmy3 * PowSrc / sum(PAmy3);
time_my3(rep,idx) = toc;

tic
for ue = 1:U
    hRDrisHSR = transpose(hRD(:,ue))*kap*diag(exp(1i*PSmy))*HSR;
    UseSig = PAmy3(ue)*(abs(hRDrisHSR*transpose(BF(ue,:)))^2);
    ItfSig = 0;
    Sig = 0;
    for itf = 1:U
        if itf ~= ue
            ItfSig = ItfSig + PAmy3(itf)*(abs(hRDrisHSR*transpose(BF(itf,:)))^2);
        end
        Sig = Sig + PAmy3(itf)*(abs(hRDrisHSR*transpose(BF(itf,:)))^2);
    end
    SNR = UseSig / (ItfSig*(1+wo(ue)) + (wo(ue)*UseSig) + sigma2);
    SNRmy2 = UseSig / (Sig*(1+wo(ue)) - UseSig + sigma2);
    SRmy3(idx,rep) = SRmy3(idx,rep) + BW*log2(1+SNR);
end
timeSR3(rep,idx) = toc;

if mod(rep,100) == 0
    fprintf('Repeat %d ',rep)
    fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
    fprintf('\n')
end

if mod(idx,100) == 0
    fprintf('Data %d ',idx)
    fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
    fprintf('\n')
end

end
end

ave_time3 = mean(mean(time_my3));
alltime3 = mean(time_my3,2);
aveSRmy3 = mean(mean(SRmy3));
allSRmy3 = mean(SRmy3,2);

fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
fprintf('\n')

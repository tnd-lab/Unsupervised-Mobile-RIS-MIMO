close all
clear
clc

fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
fprintf('\n')

PowSrc = db2pow(20-30);
BW = 50 * (10^6);
sigma2 = db2pow(-80-30);

M = 4;
N = 4;
U = 3;
B = 3;
kap = 1;
numdata = 1000;

load HSR_S3_R8_M4_N4.csv
inpHSR = HSR_S3_R8_M4_N4(2:N+1,:);
HSR = inpHSR(:,1:M).*exp(1i*inpHSR(:,M+1:M*2));

load HRD_RPG_S3_R8_M4_N4_U3_new.csv
inpHRD = HRD_RPG_S3_R8_M4_N4_U3_new(2:numdata*U+1,:);
allHRD = inpHRD(:,1:N).*exp(1i*inpHRD(:,N+1:N*2));

PSpool = wrapTo2Pi((1:2^B)*2*pi/(2^B));
numrep = 5;

numPar = 10*M*N*U;    % number of particles
numItr = 10;    % number of iterations
w = 0.9;  % inertia weight
c1 = 1.2; % particle acceleration
c2 = 1.2; % global acceleration

wo = zeros([U,1]);
for ue = 1:U
    wo(ue) = db2pow(-80-(5*ue)-30);
end

SR_PSO = zeros([numrep,numdata]);
timePSO = zeros([numrep,numdata]);

for idx = 1:numdata
hRD = transpose(allHRD(U*(idx-1)+1:U*idx,:));

for rep = 1:numrep
rng(rep)

% Initialization
Pos = rand(N+U+2*M*U,numPar);
Vel = rand(N+U+2*M*U,numPar);
BF = zeros([M*U,1]);
SRvec = zeros([numPar,1]);
SRglo = -inf;
SRitr = zeros([numItr,1]);
tic
for par = 1:numPar
    PSpos = wrapTo2Pi(2*pi*round(Pos(1:N,par)*(2^B)/(2*pi))/(2^B));
    PA = Pos(N+1:N+U,par);
    PA = PA*PowSrc/sum(PA);
    BFmag = Pos(N+U+1:N+U+M*U,par);
    BFphs = wrapTo2Pi(2*pi*Pos(N+U+M*U+1:N+U+2*M*U,par));
    com = BFmag.*exp(1i*BFphs);
    for ue = 1:U
        BF((ue-1)*M+1:(ue-1)*M+M)=com((ue-1)*M+1:(ue-1)*M+M)/norm(com((ue-1)*M+1:(ue-1)*M+M));
    end
    SRvec(par) = 0;
    for ue = 1:U
        hRDrisHSR = transpose(hRD(:,ue))*kap*diag(exp(1i*PSpos))*HSR;
        UseSig = PA(ue)*(abs(hRDrisHSR*BF((ue-1)*M+1:(ue-1)*M+M))^2);
        Sig = 0;
        for itf = 1:U
            Sig = Sig + PA(itf)*(abs(hRDrisHSR*BF((itf-1)*M+1:itf*M))^2);
        end
        SINR = UseSig / (Sig*(1+wo(ue)) - UseSig + sigma2);
        SRvec(par) = SRvec(par) + BW*log2(1+SINR);
    end
    if SRglo <= SRvec(par)
        SRglo = SRvec(par); % SR global value at 1 iteration
        PTglo = Pos(:,par); % global phase shift at 1 iteration
    end
end
SRpar = SRvec; % best SR for each particle from previous iteration
PTpar = Pos;   % best phase shift for each particle from previous iteration
SRpso = SRglo; % SR global value from all previous iterations
PTpso = PTglo; % global phase shift from all previous iterations

for itr = 1:numItr
    for par = 1:numPar
        Vel(:,par)=w*Vel(:,par)+c1*rand*(PTpar(:,par)-Pos(:,par))+c2*rand*(PTglo-Pos(:,par));
        Pos(:,par) = Pos(:,par) + Vel(:,par);
        PSpos = wrapTo2Pi(2*pi*round(Pos(1:N,par)*(2^B)/(2*pi))/(2^B));
        PA = abs(Pos(N+1:N+U,par));
        PA = PA*PowSrc/sum(PA);
        BFmag = Pos(N+U+1:N+U+M*U,par);
        BFphs = wrapTo2Pi(2*pi*Pos(N+U+M*U+1:N+U+2*M*U,par));
        com = BFmag.*exp(1i*BFphs);
        for ue = 1:U
        BF((ue-1)*M+1:ue*M)=com((ue-1)*M+1:(ue-1)*M+M)/norm(com((ue-1)*M+1:(ue-1)*M+M));
        end
        SRvec(par) = 0;
        for ue = 1:U
            hRDrisHSR = transpose(hRD(:,ue))*kap*diag(exp(1i*PSpos))*HSR;
            UseSig = PA(ue)*(abs(hRDrisHSR*BF((ue-1)*M+1:ue*M))^2);
            Sig = 0;
            for itf = 1:U
                Sig = Sig + PA(itf)*(abs(hRDrisHSR*BF((itf-1)*M+1:itf*M))^2);
            end
            SINR = UseSig / (Sig*(1+wo(ue)) - UseSig + sigma2);
            SRvec(par) = SRvec(par) + BW*log2(1+SINR);
        end
        if SRpar(par) <= SRvec(par)
            SRpar(par) = SRvec(par); % best SR for each particle from all previous iterations
            PTpar(:,par) = Pos(:,par); % best phase shift for each particle from all previous
        end
        if SRglo <= SRvec(par)
            SRglo = SRvec(par);  % SR global value at 1 iteration
            PTglo = Pos(:,par);  % global phase shift at 1 iteration
        end
    end
    if SRpso <= SRglo
        SRpso = SRglo;      % SR global value from all previous iterations
        PTpso = PTglo;      % global phase shift from all previous iterations
    end
    SRitr(itr) = SRpso;
end

SR_PSO(rep,idx) = SRpso;
timePSO(rep,idx) = toc;

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

avetimePSO = mean(mean(timePSO));
alltimePSO = mean(timePSO,2);
aveSR_PSO = mean(mean(SR_PSO));
allSR_PSO = mean(SR_PSO,2);

fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
fprintf('\n')

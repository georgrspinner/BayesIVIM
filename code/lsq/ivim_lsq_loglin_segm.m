%Fitting script for DWI/scalar IVIM: uses logarithm of data for segmented
%weighted linear least-squares (WLS) fit. 
%Features:
%-very fast linear least-squares fit
%-averaging across unique b-values with definable tolerance
%-correct weighting of data points after averaging
%-different masks/weights for different images (e.g. for artifact, void discarding in single images)
%-handling of negative values after diffusion subtraction for perfusion/IVIM estimates
%-DTI only fit if desired (if fewer than 2 data points are provided below bvalSplit for perfusion fitting)

%Input:
%bval: 1-nb (handles also nb-1)
%data: nx-ny-nb (handles also 1-1-nb; if nx>1, then ny=1 necessary)
%weight: nx-ny-(1 or nb) (number, serves as input weight/mask, can be omitted or empty: all voxels are fitted with same weight)
%pars: struct (can be omitted or empty: default values are used)

%Output:
%params: struct

%Georg Spinner, ETH Zurich & BIDMC Boston, 18.08.2019

function [paramsLoglin,paramsNonlin,paramsNonlinBiexp]=ivim_lsq_loglin_segm(bval,data,weight,pars)

%% Default fit parameters
if nargin<4; pars=struct; end; if isempty(pars); pars=struct; end
%---General
%Split b-value: bval>bvalSplit are used for diffusion fit
bvalSplit=get_options(pars,'bvalSplit',300); %can be overwritte, see below
bvalSplit=min(bvalSplit,max(bval(bval<max(bval(:)))));%Make sure there are at least 2 different b-values for D 
%Low b0 cut-off: only b-values>=b0LowCutOff are used; default: cut-off b0 image, if set to zero -> Sint will be S0
b0LowCutOff=get_options(pars,'b0LowCutOff',1); 
%Averaging of b-values: advisable because of outlier averaging and faster
averageFlag=get_options(pars,'averageFlag',true);
%Tolerance of b-value for averaging as used in Matlab round (number of digits, e.g. '2'=two digits, '-1'=integer)
averageTol=get_options(pars,'averageTol',-1); 
%Display warnings
warningDisp=get_options(pars,'warningDisp',false);
if ~warningDisp; warning('off','all'); end
%Debug mode: plot fits
debug=get_options(pars,'debug',false);
%---Loglin
%Treatment of resulting negative values after subtraction of diffusion: 'UpToFirstNeg'considers b-values up to first negative value, 'PosOnlyUpToBvalSplit' ignores negative values and ignores values from b-values>bSplit which improves precision, but can cause large bias at low SNR
perfNegMode=get_options(pars,'perfNegMode','UpToFirstNeg'); 
%---Non-linear
if ~isfield(pars,'Upper'); pars.Upper=struct; end
if ~isfield(pars,'StartPoint'); pars.StartPoint=struct; end
if ~isfield(pars,'Lower'); pars.Lower=struct; end
%Diff: high b-value
pars.Upper.S0=get_options(pars.Upper,'S0',max(abs(data(:))));
pars.StartPoint.S0=get_options(pars.StartPoint,'S0',median(abs(data(:))));
pars.Lower.S0=get_options(pars.Lower,'S0',0);
pars.Upper.D=get_options(pars.Upper,'D',2.5e-3);
pars.StartPoint.D=get_options(pars.StartPoint,'D',1e-3);
pars.Lower.D=get_options(pars.Lower,'D',0);
%Perf: all b-values
pars.Upper.f=get_options(pars.Upper,'f',1);
pars.StartPoint.f=get_options(pars.StartPoint,'f',0.1);
pars.Lower.f=get_options(pars.Lower,'f',0);
pars.Upper.DStar=get_options(pars.Upper,'DStar',50e-3);
pars.StartPoint.DStar=get_options(pars.StartPoint,'DStar',10e-3);
pars.Lower.DStar=get_options(pars.Lower,'DStar',0);


%% Pre-process data: average, cut-off b0 and vectorize
%Prepare
[bval,data]=prepare_data(bval,data); %sorting, formatting etc.
[nx,ny,nb]=size(data); %size of original data; data to fit can have different size!
if nargin<3
    weight=ones(nx,ny,nb);
end
if isempty(weight)
    weight=ones(nx,ny,nb);
end
if size(weight,3)==1
    weight=repmat(weight,[1,1,nb]); %repeat mask for every b-value
end
weight=abs(weight);
%Average
if averageFlag
    [bvalAv,dataAv,weightAv]=average_data_scalar(bval,data,weight,averageTol);
else
    bvalAv=bval;
    dataAv=data;
    weightAv=weight;
end
%Cut-off b0
bvalFit=bvalAv(bvalAv>b0LowCutOff); 
dataFit=dataAv(:,:,bvalAv>b0LowCutOff);
weightFit=weightAv(:,:,bvalAv>b0LowCutOff);
if numel(bvalFit)==1
    error('Only 1 DWI left for fitting. Fitting not possible. Check b0LowCutOff.');
end
%Data to fit: vectorize
maskFit=max(weightFit>0,[],3)>0; %logical mask of all voxels which are fitted as "maximum intensity projection" of weights along b-dimension
n=sum(maskFit(:)); %amount of voxels which are considered: averywhere where at least 1 data point is present across b-values
dataFitVec=reshape(dataFit(weightFit>0),[n,numel(bvalFit)]); %consider maximum-projection of mask, data selection for individual b-value masks comes  later
weightFitVec=reshape(weightFit(weightFit>0),[n,numel(bvalFit)]);
if prod(sum(weightFitVec,2)<2,1)
    error('Less than 2 data-points left for all voxels. Check mask.'); %highly unlikely; to prevent unexpected behavior
end

%% Fit: 1.High b-values -> D, S_int
%Loglin
[paramsLoglin.vec.SIntDiff,paramsLoglin.vec.D,paramsLoglin.vec.nDataDiff]=wls_scalar_monoexp_loglin(...
    bvalFit(bvalFit>=bvalSplit),...
    dataFitVec(:,bvalFit>=bvalSplit),...
    weightFitVec(:,bvalFit>=bvalSplit) );
%Reformatting ("de-vectorization")
paramsLoglin.SIntDiff=zeros(nx,ny); paramsLoglin.SIntDiff(maskFit)=paramsLoglin.vec.SIntDiff;
paramsLoglin.D=zeros(nx,ny); paramsLoglin.D(maskFit)=paramsLoglin.vec.D;
paramsLoglin.nDataDiff=zeros(nx,ny); paramsLoglin.nDataDiff(maskFit)=paramsLoglin.vec.nDataDiff;
paramsLoglin.bvalFit=bvalFit;
if nargout>1
    %Set fit options
    pars.fitOptionsHigh.Upper=repmat([pars.Upper.S0,-pars.Lower.D],[n,1]); %model is a*exp(+b*x)!
    pars.fitOptionsHigh.StartPoint=repmat([pars.StartPoint.S0,-pars.StartPoint.D],[n,1]); %model is a*exp(+b*x)!
    pars.fitOptionsHigh.Lower=repmat([pars.Lower.S0,-pars.Upper.D],[n,1]); %model is a*exp(+b*x)!
    %Non-lin
    [paramsNonlin.vec.SIntDiff,paramsNonlin.vec.D,paramsNonlin.vec.nDataDiff,gofNonlinHigh,fitoutputNonlinHigh]=wls_scalar_monoexp_nonlin(...
        bvalFit(bvalFit>=bvalSplit),...
        dataFitVec(:,bvalFit>=bvalSplit),...
        weightFitVec(:,bvalFit>=bvalSplit),...
        pars.fitOptionsHigh);
    %Reformatting ("de-vectorization")
    paramsNonlin.SIntDiff=zeros(nx,ny); paramsNonlin.SIntDiff(maskFit)=paramsNonlin.vec.SIntDiff;
    paramsNonlin.D=zeros(nx,ny); paramsNonlin.D(maskFit)=paramsNonlin.vec.D;
    paramsNonlin.nDataDiff=zeros(nx,ny); paramsNonlin.nDataDiff(maskFit)=paramsNonlin.vec.nDataDiff;
    paramsNonlin.residuals=zeros(nx,ny,sum(bvalFit>=bvalSplit)); 
    paramsNonlin.gofNonlinHigh=struct2D(gofNonlinHigh,maskFit);
    paramsNonlin.fitoutputNonlinHigh=struct2D(fitoutputNonlinHigh,maskFit);
    paramsNonlin.bvalFit=bvalFit;
end


%% Subtract diffusion part & get b-value indices
if sum(bvalFit<bvalSplit)<2 
    return; %if less than 2 DWI/b-values are left over, perfusion fit is not possible: output diffusion only
end
%Loglin
dataMinusDiffLoglinFitVec=dataFitVec-bsxfun(@times,exp(-paramsLoglin.vec.D*bvalFit),paramsLoglin.vec.SIntDiff);
idxBPerfPos0=dataMinusDiffLoglinFitVec>=0; %get index of positive values after subtraction
idxBPerfPos=idxBPerfPos0;
switch perfNegMode
    case 'UpToFirstNeg'
        for i=1:n
            idxBPerfPos(i,find(~idxBPerfPos0(i,:),1):end)=false; %set index of high b-value with negative values to false
        end
    case 'PosOnlyUpToBvalSplit'
        idxBPerfPos(:,bvalFit>bvalSplit)=false; %ignore b-values>bSplit, because this just adds (positive) noise, which would lead to even larger bias
end
%Nonlin
if nargout>1
    dataMinusDiffNonlinFitVec=dataFitVec-bsxfun(@times,exp(-paramsNonlin.vec.D*bvalFit),paramsNonlin.vec.SIntDiff);
end

%% Fit: 2.All/low b-values -> D*, S_int
%Loglin
[paramsLoglin.vec.SIntPerf,paramsLoglin.vec.DStar,paramsLoglin.vec.nDataPerf]=wls_scalar_monoexp_loglin(...
    bvalFit,...
    dataMinusDiffLoglinFitVec,...
    weightFitVec.*idxBPerfPos ); %exclude negative values
paramsLoglin.vec.f=paramsLoglin.vec.SIntPerf./(paramsLoglin.vec.SIntPerf+paramsLoglin.vec.SIntDiff);
paramsLoglin.vec.S0=paramsLoglin.vec.SIntDiff./(1-paramsLoglin.vec.f);
%Reformatting ("de-vectorization")
paramsLoglin.SIntPerf=zeros(nx,ny); paramsLoglin.SIntPerf(maskFit)=paramsLoglin.vec.SIntPerf;
paramsLoglin.DStar=zeros(nx,ny); paramsLoglin.DStar(maskFit)=paramsLoglin.vec.DStar;
paramsLoglin.nDataPerf=zeros(nx,ny); paramsLoglin.nDataPerf(maskFit)=paramsLoglin.vec.nDataPerf;
paramsLoglin.f=zeros(nx,ny); paramsLoglin.f(maskFit)=paramsLoglin.vec.f;
paramsLoglin.S0=zeros(nx,ny); paramsLoglin.S0(maskFit)=paramsLoglin.vec.S0;
if nargout>1
    %Set fit options
    pars.fitOptionsAll.Upper=[paramsNonlin.vec.SIntDiff.*pars.Upper.f./(1-pars.Upper.f),-repmat(pars.Lower.DStar,[n,1])]; %model is a*exp(+b*x)!
    pars.fitOptionsAll.StartPoint=[paramsNonlin.vec.SIntDiff.*pars.StartPoint.f./(1-pars.StartPoint.f),-repmat(pars.StartPoint.DStar,[n,1])]; %model is a*exp(+b*x)!
    pars.fitOptionsAll.Lower=[paramsNonlin.vec.SIntDiff.*pars.Lower.f./(1-pars.Lower.f),-repmat(pars.Upper.DStar,[n,1])]; %model is a*exp(+b*x)!
    %Nonlin
    [paramsNonlin.vec.SIntPerf,paramsNonlin.vec.DStar,paramsNonlin.vec.nDataPerf,gofNonlinAll,fitoutputNonlinAll]=wls_scalar_monoexp_nonlin(...
        bvalFit,...
        dataMinusDiffNonlinFitVec,...
        weightFitVec,...
        pars.fitOptionsAll); %also include negative values
    paramsNonlin.vec.f=paramsNonlin.vec.SIntPerf./(paramsNonlin.vec.SIntPerf+paramsNonlin.vec.SIntDiff);
    paramsNonlin.vec.S0=paramsNonlin.vec.SIntDiff./(1-paramsLoglin.vec.f);
    %Reformatting ("de-vectorization")
    paramsNonlin.SIntPerf=zeros(nx,ny); paramsNonlin.SIntPerf(maskFit)=paramsNonlin.vec.SIntPerf;
    paramsNonlin.DStar=zeros(nx,ny); paramsNonlin.DStar(maskFit)=paramsNonlin.vec.DStar;
    paramsNonlin.nDataPerf=zeros(nx,ny); paramsNonlin.nDataPerf(maskFit)=paramsNonlin.vec.nDataPerf;
    paramsNonlin.f=zeros(nx,ny); paramsNonlin.f(maskFit)=paramsNonlin.vec.f;
    paramsNonlin.S0=zeros(nx,ny); paramsNonlin.S0(maskFit)=paramsNonlin.vec.S0;
    paramsNonlin.gofNonlinAll=struct2D(gofNonlinAll,maskFit);
    paramsNonlin.fitoutputNonlinAll=struct2D(fitoutputNonlinAll,maskFit);
end

%% Bi-exp nonlin fit
if nargout>2
    %Set fit options
    pars.fitOptionsBiexp.Upper=repmat([pars.Upper.D,pars.Upper.DStar,pars.Upper.S0,pars.Upper.f],[n,1]);
    pars.fitOptionsBiexp.StartPoint=repmat([pars.StartPoint.D,pars.StartPoint.DStar,pars.StartPoint.S0,pars.StartPoint.f],[n,1]);
    pars.fitOptionsBiexp.Lower=repmat([pars.Lower.D,pars.Lower.DStar,pars.Lower.S0,pars.Lower.f],[n,1]);
    pars.fitOptionsBiexp.StartPoint=[paramsNonlin.vec.D,paramsNonlin.vec.DStar,paramsNonlin.vec.S0,paramsNonlin.vec.f];
    %Fit
    [paramsNonlinBiexp.vec.S0,paramsNonlinBiexp.vec.D,paramsNonlinBiexp.vec.f,paramsNonlinBiexp.vec.DStar,paramsNonlinBiexp.vec.nDataUsed,gofNonlinBiexp,fitoutputNonlinBiexp]=wls_scalar_biexp_nonlin(...
        bvalFit,...
        dataFitVec,...
        weightFitVec,...
        pars.fitOptionsBiexp); %also include negative values
    %Reformatting ("de-vectorization")
    paramsNonlinBiexp.D=zeros(nx,ny); paramsNonlinBiexp.D(maskFit)=paramsNonlinBiexp.vec.D;
    paramsNonlinBiexp.DStar=zeros(nx,ny); paramsNonlinBiexp.DStar(maskFit)=paramsNonlinBiexp.vec.DStar;
    paramsNonlinBiexp.nDataUsed=zeros(nx,ny); paramsNonlinBiexp.nDataUsed(maskFit)=paramsNonlinBiexp.vec.nDataUsed;
    paramsNonlinBiexp.f=zeros(nx,ny); paramsNonlinBiexp.f(maskFit)=paramsNonlinBiexp.vec.f;
    paramsNonlinBiexp.S0=zeros(nx,ny); paramsNonlinBiexp.S0(maskFit)=paramsNonlinBiexp.vec.S0;
    paramsNonlinBiexp.gof=struct2D(gofNonlinBiexp,maskFit);
    paramsNonlinBiexp.fitoutput=struct2D(fitoutputNonlinBiexp,maskFit);
    paramsNonlinBiexp.bvalFit=bvalFit;
end

if ~warningDisp; warning('on','all'); end
end


%% Data preparation
function [bval,data]=prepare_data(bval,data)
[bval,idx_bval]=sort(bval,'ascend'); %sort bval
bval=double(bval);
data=double(data);
if iscolumn(bval)
    bval=bval';
end
if isvector(data)==1 
    data=squeeze(data);
    data=permute(data,[2,3,1]); %make into 1-1-nb
end
data=data(:,:,idx_bval); %sort data
if size(data,3)~=numel(bval)
    error('Number of b-values different from number of images');
end
end

%% Function for handling of default struct values
function valueTmp = get_options(structIn, structField, valueTmp, obligatory)
if nargin<4
    obligatory = 0;
end
if isfield(structIn, structField)
    valueTmp = eval(['structIn.' structField ';']);
elseif obligatory
    error(['You have to provide options.' structField '.']);
end 
end

%% Averaging: magnitude only, assumes b-values in ascending order
function [bvalAv,dataAv,weightAv]=average_data_scalar(bval,data,weight,averageTol)
data=abs(data);
nB=numel(bval);
bvalRounded=round(bval,averageTol);
[bvalUnique,subBvalUnique]=unique(bvalRounded);
nBAv=numel(bvalUnique);
bvalAv=zeros(1,nBAv);
dataAv=zeros(size(data,1),size(data,2),nBAv);
weightAv=ones(size(data,1),size(data,2),nBAv);
for i=1:nBAv
    if i<nBAv
        idxToAvTmp=subBvalUnique(i):subBvalUnique(i+1)-1;
    else
        idxToAvTmp=subBvalUnique(i):nB;
    end
    bvalAv(1,i)=mean(bval(idxToAvTmp));
    dataAv(:,:,i)=mean(data(:,:,idxToAvTmp),3);
    weightAv(:,:,i)=sum(weight(:,:,idxToAvTmp),3); %maskAv is true if there is at least one true in the averaged masks
end
end

%% Convert vector struct to 2D: dimensions of fields limited to 3, ignores non-numeric fields 
function out=struct2D(in,mask)
fieldnamesIn=fieldnames(in);
[nx,ny]=size(mask);
n=sum(mask(:));
out=struct;
for i=1:numel(fieldnamesIn)
    if isnumeric(eval(strcat('in(1).',fieldnamesIn{i})))
        eval(strcat('sizeAdd=size(','in(1).',fieldnamesIn{i},');'));
        eval(strcat('out.',fieldnamesIn{i},'=zeros([nx,ny,sizeAdd]);'));
        temp=zeros([n,sizeAdd]);
        for j=1:size(in,2)
            eval(strcat('temp(j,:,:)=','in(j).',fieldnamesIn{i},';'));
        end
        eval(strcat('out.',fieldnamesIn{i},'(repmat(mask,[1,1,sizeAdd]))=temp;'));
    end
end
end

%% Least-squares solver for logarithm of data
% Dimension of weights n-nb: serves as both weights for WLS and for data
% selection (i.e. only voxels with weights>0 are fitted)
function [SInt,D,nDataUsed]=wls_scalar_monoexp_loglin(bval,data,weights)
n=size(data,1);
logData=log(data); 
if nargin<3
    weights=ones(n,1);
end
nDataUsed=zeros(n,1);
idxDataToFit=weights>0;
x=zeros(n,2);
%Check if fast inversion of whole data matrix is possible: weights must be
%the same for all voxels across b-values
fastFlag=true;
weights1=weights(1,:);
i=1;
while fastFlag && i<=n
    fastFlag=fastFlag & prod(weights1==weights(i,:));
    i=i+1;
end
%WLS
if fastFlag %fast: all data points in matrix
    nDataUsed(:)=sum(weights(1,:),2); %same for all voxels, hence choose first one
    A=cat(2,ones(sum(idxDataToFit(1,:)),1),bval(idxDataToFit(1,:))'); 
    B=logData(:,idxDataToFit(1,:))';
    w=weights(1,idxDataToFit(1,:))'; %same for all voxels, hence choose first one
    x = (w.*A)\(w.*B); %weighted least squares (WLS); lscov is alternative; OLS if weights are all one
    x=x';
else %slow: loop over voxels
    for i=1:n
        nDataUsed(i)=sum(weights(i,:),2);
        if sum(idxDataToFit(i,:))< 2
            continue;
        else
            A=cat(2,ones(sum(idxDataToFit(i,:)),1),bval(idxDataToFit(i,:))');
            B=logData(i,idxDataToFit(i,:))';
            w=weights(i,idxDataToFit(i,:))';
            x(i,:) = (w.*A)\(w.*B); %weighted least squares (WLS); lscov is alternative; OLS if weights are all one
        end
    end
end
SInt=exp(x(:,1));
D=-x(:,2);
end

function [SInt,D,nDataUsed,gof,fitoutput]=wls_scalar_monoexp_nonlin(bval,data,weights,fitOptions)
n=size(data,1);
if nargin<3
    weights=ones(n,1);
end
nDataUsed=zeros(n,1);
idxDataToFit=weights>0;
SInt=zeros(n,1); D=zeros(n,1);
%WLS: nonlin
fitOptionsTemp=fitoptions('exp1');
for i=1:n
    %i
    nDataUsed(i)=sum(weights(i,:),2);
    if sum(idxDataToFit(i,:))< 2
        continue;
    else
        %Get fitoptions
        fitOptionsTemp.Upper=fitOptions.Upper(i,:);
        fitOptionsTemp.StartPoint=fitOptions.StartPoint(i,:);
        fitOptionsTemp.Lower=fitOptions.Lower(i,:);
        fitOptionsTemp.Weights=weights(i,idxDataToFit(i,:))';
        if nargout<=3
            fitobject = fit(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))','exp1',fitOptionsTemp);
        else
            [fitobject,gof(i),fitoutput(i)] = fit(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))','exp1',fitOptionsTemp);
        end
        SInt(i)=fitobject.a;
        D(i)=-fitobject.b;
        %Plot: debugging
        %figure; semilogy(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))','b.-'); hold on;
        %semilogy(bval(idxDataToFit(i,:))',SInt(i)*exp(-D(i)*idxDataToFit(i,:))','k--'); hold off; drawnow;
    end
end
end

function [S0,D,f,DStar,nDataUsed,gof,fitoutput]=wls_scalar_biexp_nonlin(bval,data,weights,fitOptions)
n=size(data,1);
if nargin<3
    weights=ones(n,1);
end
nDataUsed=zeros(n,1);
idxDataToFit=weights>0;
S0=zeros(n,1); D=zeros(n,1); f=zeros(n,1); DStar=zeros(n,1);
%WLS: nonlin
fitType=fittype('S0*(f*exp(-b*DStar) + (1-f)*exp(-b*D))',...
    'independent','b'); %IVIM model, variable ordering is D,DStar,S0,f,b
fitOptionsTemp=fitoptions(fitType); 
for i=1:n
    %i
    nDataUsed(i)=sum(weights(i,:),2);
    if sum(idxDataToFit(i,:))< 2
        continue;
    else
        %Get fitoptions
        fitOptionsTemp.Upper=fitOptions.Upper(i,:);
        fitOptionsTemp.StartPoint=fitOptions.StartPoint(i,:);
        fitOptionsTemp.Lower=fitOptions.Lower(i,:);
        fitOptionsTemp.Weights=weights(i,idxDataToFit(i,:))';
        %[fitobject,gof,output] =
        %fit(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))','exp1',pars.fitOptionsHigh);
        %%full output for debugging;
        if nargout<=5
            [fitobject] = fit(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))',fitType,fitOptionsTemp);
        else
            [fitobject,gof(i),fitoutput(i)] = fit(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))',fitType,fitOptionsTemp);
        end
        D(i)=fitobject.D;
        DStar(i)=fitobject.DStar;
        S0(i)=fitobject.S0;
        f(i)=fitobject.f;
        %Plot: debugging
        %h=figure; %DEBUG
        %semilogy(bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))','b.-'); hold on;
        %plot(fitobject,bval(idxDataToFit(i,:))',data(i,idxDataToFit(i,:))'); %ylim([0.05,0.35]); 
        %drawnow;
        %set(gca, 'YScale', 'log');
        %semilogy(bval(idxDataToFit(i,:))',SInt(i)*exp(-D(i)*idxDataToFit(i,:))','k--'); hold off; drawnow;
    end
end
end



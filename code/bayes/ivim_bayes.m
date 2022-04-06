%Input: 
%-bval: row or column vector of length nb
%-data: images of size nx x ny x nb
%-mask: logical image of size nx x ny; background must be removed to get
%sensible results
%-pars: parameter struct; defaults given below. If the spatial prior is
%used, hyper-parameters for the hierarchical Normal prior (pars.muFix and
%pars.sigmaFix) must be given as input arguments.

%Outputs:
%Note that variables containing suffix "All" contain all MCMC samples and
%are hence huge! Thus, those output arguments should only be called if
%necessary.

%Copyright: Georg Ralph Spinner, ETH Zurich, 22.11.2021


function [params,theta,mu,sigma,thetaAll,muAll,sigmaAll,stdw,indicatorSum]=ivim_bayes(bval,data,mask,pars)

%% Prepare input
if nargin<4; pars=struct; end; if isempty(pars); pars=struct; end
if nargin<3
	mask=true(size(data,1),size(data,2));
else
	if isempty(mask)
		mask=true(size(data,1),size(data,2));
	end
end
%% Prepare data
[bval,data]=prepare_data(bval,data); %sorting, formatting etc.
[nx,ny,nb]=size(data); %size of original data;
n=sum(mask(:));
%Data to fit: vectorize
dataVec=zeros(n,nb);
for b=1:nb
	temp=data(:,:,b);
	dataVec(:,b)=abs(double(temp(mask)));
end
dataVec(isinf(dataVec))=0; dataVec(isnan(dataVec))=0;

%% Default parameters
%chains
pars.nburnin=get_options(pars,'nburnin',1000); %burn in samples
pars.ndiscard=get_options(pars,'ndiscard',0); %discarded samplesa fter burn-in
pars.nsamples=get_options(pars,'nsamples',2500); %used samples 
pars.adaptStepsize=get_options(pars,'adaptStepsize',50); %alternative: 100
pars.nchains=get_options(pars,'nchains',4); %number of Markov chains
pars.doparallel=get_options(pars,'doparallel',true); %parallel computing
pars.overdisp=get_options(pars,'overdisp',0.01); %relative dispersion of start values if multiple Markov chains are used 
%spatial prior
pars.spatialPriorFlag=get_options(pars,'spatialPriorFlag',false); %usage of spatial prior (relatively slow); requires setting of pars.muFix and pars.sigmaFix
pars.blockAddSize=get_options(pars,'blockAddSize',2); %neighborhood size is 2*blockAddSize + 1 (e.g. 2*1 + 1 = 3 -> 3x3 neighb.)
pars.tauSpatial=get_options(pars,'tauSpatial',2); %regularization strength of spatial prior (2 corresponds to 1 in the MEDIMA paper)
pars.SNR=get_options(pars,'SNR',1); %for (voxel-wise) adaption to different SNR of smoothing; value of "1" has no effect
%hierarchical prior
pars.priorChoice=get_options(pars,'priorChoice','normal+jp'); %posibilities: 'uniform' 'normal+jp' 'normal'
pars.paramTransform=get_options(pars,'paramTransform','log'); %posibilities: 'log' 'no'
%likelihood segmentation
pars.segmFlag=get_options(pars,'segmFlag',true); %usage of likelihood segmentation
pars.bsplit=get_options(pars,'bsplit',800); %splitting b-value - needs adaptation according to application!
%start value and constraints for uniform prior (specifically)
pars.pL=get_options(pars,'pL',[0,0,0]); %only for uniform prior
pars.pU=get_options(pars,'pU',[2.5e-3,1,50e-3]); %only for uniform prior
pars.pStart=get_options(pars,'pStart',repmat([1e-3,0.1,10e-3],[n,1]).*(1+0.1*randn(n,3))); %can get stuck during adaptation!
%pars.pStart=get_options(pars,'pStart',repmat([0.75e-3,0.065,6.7e-3],[n,1]).*(1+0.1*randn(n,3)));

%% 
if pars.spatialPriorFlag && strcmp(pars.priorChoice,'normal+jp')
	error('Combination of spatial prior and hierarchical prior with Jeffreys hyper-prior not implemented.');
end


%% parameter transformations
switch pars.paramTransform
	case 'log'
		p2t=@(p) cat(2,log(p(:,1)),log(p(:,2))./(1-p(:,2)),log(p(:,3)));
		t2p=@(t) cat(2,exp(t(:,1)),1./(1+exp(-t(:,2))),exp(t(:,3)));
	case 'no'
		p2t=@(p) p; %dummy
		t2p=@(t) t; %dummy
end

%% Set start point
thetaStart=p2t(pars.pStart);
pars.sigmaStart=get_options(pars,'sigmaStart',cov(thetaStart));
pars.muStart=get_options(pars,'muStart',mean(thetaStart,1));
muStart=pars.muStart; %not used for flat prior
sigmaStart=pars.sigmaStart; %not used for flat prior

%% Set fixed values for normal prior
pars.muFix=get_options(pars,'muFix',median(p2t(pars.pStart),1)); %taken from data if not given
pars.sigmaFix=get_options(pars,'sigmaFix',cov(p2t(pars.pStart),1)); %taken from data if not given

%% MCMC sampling
% %Check burnin time (sequence: burnin with adaption->discard, burnin no adaptation->discard, actual samples)
% if  pars.nburnin>pars.nsamples %all samples after adaptation would be used
% pars.nsamples=pars.nsamples+(pars.nburnin-pars.nsamples); %sampling time prolonged to allow for actual sampling (see "Prepare Output")
% end
%Allocate memory
thetaAll=zeros(pars.nchains,pars.nburnin+pars.ndiscard+pars.nsamples,n,3);
muAll=zeros(pars.nchains,pars.nburnin+pars.ndiscard+pars.nsamples,3);
sigmaAll=zeros(pars.nchains,pars.nburnin+pars.ndiscard+pars.nsamples,3,3);
stdw=zeros(pars.nchains,n,3); %only last iteration
indicatorSum=zeros(pars.nchains,n,3); %only last iteration
%Prevent variable broadcasting
nburnin=pars.nburnin;
ndiscard=pars.ndiscard;
nsamples=pars.nsamples;
adaptStepsize=pars.adaptStepsize;
segmFlag=pars.segmFlag;
bsplit=pars.bsplit;
priorChoice=pars.priorChoice;
spatialPriorFlag=pars.spatialPriorFlag;
blockAddSize=pars.blockAddSize;
tauSpatial=pars.tauSpatial;
SNR=pars.SNR;
pL=pars.pL; pU=pars.pU;
overdispFac=ones(pars.nchains,1); %default is just one
muFix=pars.muFix;
sigmaFix=pars.sigmaFix;
if pars.nchains>1
	for i=1:pars.nchains
		overdispFac(i)=1+pars.overdisp*( i*(2/(pars.nchains-1)) - ((pars.nchains+1)/(pars.nchains-1)) );
	end
end
if ~pars.doparallel
	for i=1:pars.nchains
		[thetaAll(i,:,:,:),muAll(i,:,:),sigmaAll(i,:,:,:),stdw(i,:,:),indicatorSum(i,:,:)]=...
			metropolis_hastings(bval,dataVec,...
			thetaStart*overdispFac(i),muStart*overdispFac(i),sigmaStart*overdispFac(i),...
			nburnin,ndiscard,nsamples,adaptStepsize,...
			priorChoice,segmFlag,bsplit,...
			spatialPriorFlag,blockAddSize,mask,tauSpatial,SNR,...
			p2t, t2p,...
			pL,pU,...
			muFix,sigmaFix);
	end
else
	%Set-up parallel pool
	p=gcp('nocreate');
	if isempty(p)
% 		if isunix
% 			nworker=4;
% 		elseif ispc
% 			nworker=2;
% 		end
        nworker=maxNumCompThreads; %use all cores
		ppool=parpool('local',nworker);
	end
	
	parfor i=1:pars.nchains
		[thetaAll(i,:,:,:),muAll(i,:,:),sigmaAll(i,:,:,:),stdw(i,:,:),indicatorSum(i,:,:)]=...
			metropolis_hastings(bval,dataVec,...
			thetaStart*overdispFac(i),muStart*overdispFac(i),sigmaStart*overdispFac(i),...
			nburnin,ndiscard,nsamples,adaptStepsize,...
			priorChoice,segmFlag,bsplit,...
			spatialPriorFlag,blockAddSize,mask,tauSpatial,SNR,...
			p2t, t2p,...
			pL,pU,...
			muFix,sigmaFix);
	end
end


%% Prepare Output & calculate RHat
switch pars.paramTransform
	case 'log'
		t2pOutputFun={@(x) exp(x),@(x) exp(x)./(1+exp(x)), @(x) exp(x)};
	case 'no'
		t2pOutputFun={@(x) x,@(x) x, @(x) x}; %dummy
end

% %uses lot of RAM
% for p=1:3
%     [mu.Rhat(p),mu.mean(p),mu.std(p),mu.median(p)]=EPSR(muAll(:,nburnin+ndiscard+1:end,p)');
%     for p1=1:3
%         [sigma.Rhat(p,p1),sigma.mean(p,p1),sigma.std(p,p1),sigma.median(p,p1)]=EPSR(sigmaAll(:,nburnin+ndiscard+1:end,p,p1)');
%     end
%     for i=1:n
%         [theta.Rhat(i,p),meanTmp(i,p),stdTmp(i,p),medianTmp(i,p)]=EPSR(thetaAll(:,nburnin+ndiscard+1:end,i,p)');
%     end
% end

%uses less RAM, but takes a bit longer
%Rhat is calculated as in EPSR function, but explicitly coded here
%to avoid copying the large arrays of samples
idxSample=nburnin+ndiscard+1 : nburnin+ndiscard+nsamples;
nsamples=numel(idxSample);
m=pars.nchains;



%mu
for p=1:3
    mu.mean(p)=mean(mean(muAll(:,idxSample,p)));
    mu.median(p)=median(reshape(muAll(:,idxSample,p),[],1),1);
    mu.std(p)=std(reshape(muAll(:,idxSample,p),[],1),[],1);
    %Rhat
    W = (1/m)*sum( var(muAll(:,idxSample,p),[],2) );
    vhat = ((nsamples-1)/nsamples)*(1/m)*sum( var(muAll(:,idxSample,p),[],2) ) +...
        (1/nsamples)*(nsamples/(m-1))*sum( (mean(muAll(:,idxSample,p),2)-mu.mean(p)).^2);
    mu.Rhat(p) = sqrt(vhat/(W+eps));
end

%sigma
for p=1:3
    for p1=1:3
        sigma.mean(p,p1)=mean(mean(sigmaAll(:,idxSample,p,p1)));
        sigma.median(p,p1)=median(reshape(sigmaAll(:,idxSample,p,p1),[],1),1);
        sigma.std(p,p1)=std(reshape(sigmaAll(:,idxSample,p,p1),[],1),[],1);
        %Rhat
        W = (1/m)*sum( var(sigmaAll(:,idxSample,p,p1),[],2) );
        vhat = ((nsamples-1)/nsamples)*(1/m)*sum( var(sigmaAll(:,idxSample,p,p1),[],2) ) +...
            (1/nsamples)*(nsamples/(m-1))*sum( (mean(sigmaAll(:,idxSample,p,p1),2)-sigma.mean(p,p1)).^2);
        sigma.Rhat(p,p1) = sqrt(vhat/(W+eps));
    end
end

%theta
theta.mean=zeros(n,3);
theta.median=zeros(n,3);
theta.std=zeros(n,3);
theta.Rhat=zeros(n,3);
for p=1:3
    for i=1:n
        theta.mean(i,p)=mean(mean(thetaAll(:,idxSample,i,p)));
        theta.median(i,p)=median(reshape(thetaAll(:,idxSample,i,p),[],1),1);
        theta.std(i,p)=std(reshape(thetaAll(:,idxSample,i,p),[],1),[],1);
        %Rhat
        W = (1/m)*sum( var(thetaAll(:,idxSample,i,p),[],2) );
        vhat = ((nsamples-1)/nsamples)*(1/m)*sum( var(thetaAll(:,idxSample,i,p),[],2) ) +...
            (1/nsamples)*(nsamples/(m-1))*sum( (mean(thetaAll(:,idxSample,i,p),2)-theta.mean(i,p)).^2);
        theta.Rhat(i,p) = sqrt(vhat/(W+eps));
    end
end

%params
params.mean=zeros(n,3);
params.median=zeros(n,3);
params.std=zeros(n,3);
params.Rhat=zeros(n,3);
for p=1:3
    %params.mean(:,:,p)=vec2im(t2pOutputFun{p}(theta.mean(:,p)),mask);
    %params.median(:,:,p)=vec2im(t2pOutputFun{p}(theta.median(:,p)),mask);
    for i=1:n
        paramsTemp=t2pOutputFun{p}(thetaAll(:,idxSample,i,p));
        params.mean(i,p)=mean(reshape(paramsTemp,[],1),1);
        params.median(i,p)=median(reshape(paramsTemp,[],1),1);
        params.std(i,p)=std(reshape(paramsTemp,[],1),[],1);
        
        %Rhat
        W = (1/m)*sum( var( paramsTemp,[],2) );
        vhat = ((nsamples-1)/nsamples)*(1/m)*sum( var( paramsTemp,[],2) ) +...
            (1/nsamples)*(nsamples/(m-1))*sum( (mean(paramsTemp,2)-params.mean(i,p)).^2);
        params.Rhat(i,p) = sqrt(vhat/(W+eps));
    end
end
%-> 2D
params.mean=reshape(vec2im(params.mean,repmat(mask,[1,1,3])),[nx,ny,3]);
params.median=reshape(vec2im(params.median,repmat(mask,[1,1,3])),[nx,ny,3]);
params.std=reshape(vec2im(params.std,repmat(mask,[1,1,3])),[nx,ny,3]);
params.Rhat=reshape(vec2im(params.Rhat,repmat(mask,[1,1,3])),[nx,ny,3]);


end

function [theta,mu,sigma,stdw,indicatorSum]=metropolis_hastings(bval,dataVec,...
	thetaStart,muStart,sigmaStart,...
	nburnin,ndiscard,nsamples,adaptStepsize,...
	priorChoice,segmFlag,bsplit,...
	spatialPriorFlag,blockAddSize,mask,tauSpatial,SNR,...
	p2t, t2p,...
	pL,pU,...
	muFix,sigmaFix)

%% General
warning('off','all');

%% Define necessary functions
%basics
nb=numel(bval);
nbsplit=find(bval==bsplit,1);

%likelihood & model
ivimFun=@(p,b,nb) repmat(p(:,2),[1 nb]).*exp(-p(:,3)*b) + ...
	(1-repmat(p(:,2),[1 nb])).*exp(-p(:,1)*b);
dwiFun=@(p,b,nb) (1-repmat(p(:,2),[1 nb])).*exp(-p(:,1)*b);
segmFun=@(p) cat(2, ivimFun(p,bval(1:nbsplit-1),nbsplit-1), dwiFun(p,bval(nbsplit:nb),(nb-nbsplit+1)) );
if segmFlag==false
	modelFun=@(p) ivimFun(p,bval,nb); %BSP Orton
else
	modelFun=@(p) segmFun(p); %Segmented
end
likelihoodFun=@(g) (sum(dataVec.*dataVec,2) - (sum(dataVec.*g,2).^2)./(sum(g.*g,2))).^(-nb/2);
%Metropolis Hastings part: likelihood
alphaLikelihoodFun= @(thetaProp,thetaCurr) ...
	likelihoodFun(modelFun(t2p(thetaProp)))./...
	likelihoodFun(modelFun(t2p(thetaCurr)));

%prior
switch priorChoice
	case 'normal+jp'
		doGibbs=true;
		%priorFun=@(theta,mu,sigma) mvnpdf(theta,mu,sigma); %if sigma is not SPD -> error (frequent problem)
		priorHierarchFun=@(theta,mu,sigma,sigmaDet,sigmaInv) 1/realsqrt(abs(2*pi*sigmaDet))*...
			exp(-1/2*sum( ((theta-repmat(mu,[size(theta,1),1]))*sigmaInv).*(theta-repmat(mu,[size(theta,1),1])),2) );
		
		
	case 'normal'
		doGibbs=false;
		sigmaFixDet=det(sigmaFix);
		sigmaFixInv=inv(sigmaFix);
		%sigma, sigmaDet, sigmaInv are dummy arguments - only there for
		%consistency
		priorHierarchFun=@(theta,mu,sigma,sigmaDet,sigmaInv) 1/realsqrt(abs(2*pi*sigmaFixDet))*...
			exp(-1/2*sum( ((theta-repmat(muFix,[size(theta,1),1]))*sigmaFixInv).*(theta-repmat(muFix,[size(theta,1),1])),2) );
		
		
	case 'uniform'
		doGibbs=false;
		thetaL=p2t(pL);
		thetaL(isnan(thetaL))=-Inf; %can lead to excessive exploration!
		thetaU=p2t(pU);
		thetaU(isnan(thetaU))=Inf;  %can lead to excessive exploration!
		priorHierarchFun=@(theta,mu,sigma,sigmaDet,sigmaInv)...
			prod( (theta>thetaL) & (theta<thetaU), 2);%.* (thetaHigh<=theta(:,2));% .*...
		%Metropolis Hastings part: hierarchical prior
		%prod(~isnan(theta),2) .*...
		%prod(~isinf(theta),2);
		%skipHyperpriorFlag=true;
		
		
end
%Metropolis Hastings part: hierarchical prior
alphaPriorHierarchFun = @(thetaProp,thetaCurr,mu,sigma,sigmaDet,sigmaInv)...
	priorHierarchFun( thetaProp,mu,sigma,sigmaDet, sigmaInv ) ./ priorHierarchFun( thetaCurr,mu,sigma,sigmaDet, sigmaInv  );

if ~spatialPriorFlag
	%wSumNeighbFun=@(theta,thetaPrev) 1; %dummy
	%spatialPriorFun=@(wSumNeighbVal) 1; %dummy
	alphaPriorSpatialFun=@(thetaProp,thetaCurr,Neighb,weights) 1; %dummy
else
	%Neighborhood calculation
	%blockSizeTot=(blockAddSize*2+1)^2;
	%blockCenterIdx=floor(blockSizeTot/2)+1;
	[rMask,cMask]=find(mask);
	idxMaskBoxX=[min(rMask),max(rMask)]; %leads to crash if distance from outermost pixel in mask from image border is smaller than neighborhood!
	idxMaskBoxY=[min(cMask),max(cMask)];
	wSumNeighbFun=@(theta,thetaNeighb,weights) ...
		wSumNeighb(theta,thetaNeighb,weights,mask,idxMaskBoxX,idxMaskBoxY,blockAddSize);
	spatialPriorFun=@(wSumNeighbVal)...
		exp( (-1/tauSpatial) * wSumNeighbVal ); 
	alphaPriorSpatialFun =@(thetaProp,thetaCurr,thetaNeighb,weights)...
		spatialPriorFun(wSumNeighbFun(thetaProp,thetaNeighb,weights)) ./ ...
		spatialPriorFun(wSumNeighbFun(thetaCurr,thetaNeighb,weights));
end

% %Metropolis Hastings part: combine likelihood & hierarchical prior ratios,
% %spatial prior is added later to avoid expensive recalculations
% alphaFun=@(thetaProp,thetaCurr,thetaPrev,mu,sigma, sigmaDet, sigmaInv)...
% 	alphaLikelihoodFun(thetaProp,thetaCurr) .*...
% 	alphaPriorHierarchFun(thetaProp,thetaCurr,mu,sigma,sigmaDet,sigmaInv) .*...
% 	alphaPriorSpatialFun(thetaProp,thetaCurr,weights);


%% Prepare
n=size(dataVec,1);
mu=zeros(nburnin+ndiscard+nsamples,3);
sigma=zeros(nburnin+ndiscard+nsamples,3,3);
theta=zeros(nburnin+ndiscard+nsamples,n,3);

%Initialize
theta(1,:,:)=thetaStart;
%stdw=0.1*ones(n,3);
%stdw=cat(2,0.1e-3*ones(n,1),0.01*ones(n,1),1e-3*ones(n,1));
stdw=ones(n,3);
for p=1:3
	stdw(:,p)=iqr(thetaStart(:,p));
	%stdw(:,p)=abs(thetaStart(:,p)); %do not use
end
indicatorSum=zeros(n,3);
switch priorChoice
	case 'normal+jp'
		mu(1,:,:)=muStart;
		sigma(1,:,:)=sigmaStart;
	case 'normal'
		mu=repmat(muFix,[nburnin+ndiscard+nsamples,1]);
		sigma=repmat(permute(sigmaFix,[3,1,2]),[nburnin+ndiscard+nsamples,1,1]);
		sigmaDet=zeros(3,3); %dummy, because no Gibbs sampling
		sigmaInv=zeros(3,3); %dummy, because no Gibbs sampling
	case 'uniform'
		%leave mu & sigma at zero, they are not used
        sigmaDet=zeros(3,3); %dummy, because no Gibbs sampling
		sigmaInv=zeros(3,3); %dummy, because no Gibbs sampling
end

for j=2:nburnin+ndiscard+nsamples
	%j
	if mod(j,adaptStepsize)==0
		%keyboard
        if j<=nburnin
            strDisp='adapting';
        elseif j<=nburnin+ndiscard
            strDisp='discarding';
        elseif  j<=nburnin+ndiscard+nsamples
            strDisp='sampling';
        end
		%fprintf('%d / %d \n',round(j),round(nburnin+ndiscard+nsamples))
		disp([num2str(j,'%d'),'/',num2str(nburnin+ndiscard+nsamples,'%d'),...
            ' - ',strDisp,' - ',...
            'acceptance: ',...
			num2str(median(indicatorSum(:,1)/adaptStepsize),'%.2f'),'/',...
			num2str(median(indicatorSum(:,2)/adaptStepsize),'%.2f'),'/',...
			num2str(median(indicatorSum(:,3)/adaptStepsize),'%.2f'),' - ',....
			'stdw: ',...
			num2str(median(stdw(:,1)),'%.2f'),'/',...
			num2str(median(stdw(:,2)),'%.2f'),'/',...
			num2str(median(stdw(:,3)),'%.2f')])
		
	end
	
	
	%% Sample from hyper-priors
	if doGibbs
		%%mu: Normal
		sigmaMean=1/n*cov(squeeze(theta(j-1,:,:)));
		%mu(j,:)=mvnrndFun(mean(squeeze(theta(j-1,:,:)),1)',sigmaMean,sigmaDetFun(sigmaMean),sigmaInvFun(sigmaMean));
		%%does not work
		
		%mu(j,:)=mvnrndFun(mean(squeeze(theta(j-1,:,:)),1),sigmaMean);
		mu(j,:)=mvnrnd(mean(squeeze(theta(j-1,:,:)),1),sigmaMean);
		thetaMinusMu=squeeze(theta(j-1,:,:))-repmat(mu(j,:),[n,1]); %n x 3
		
		%sigma: inverse Wishart
		sigma(j,:,:)=iwishrnd( thetaMinusMu' * thetaMinusMu, n-3);
		%sigma(j,:,:)=iwishrndFun( 10*eye(3) + thetaMinusMu' * thetaMinusMu, n-3); %TEST
		
		sigmaDet=det(squeeze(sigma(j,:,:))); %for Metropolis Hastings for nromal + JP (pre-calculation)
		sigmaInv=inv(squeeze(sigma(j,:,:)));
	end
	
	%%  Sample parameters
	%All voxels
	%for k=1:n
	thetaPrev=squeeze(theta(j-1,:,:)); %previous estimates from all parameters: only needed for spatial prior weights
	thetaCurr=thetaPrev; %current estimates: initialize with old estimates (j-1) change/propose new value for one parameter after the other
	thetaProp=thetaCurr;
	for p=1:3
		%pCurr=t2p(thetaCurr); %if p>1: variables contains updates from ther ps (i.e. 1,2)
		%theta(j,:,p)=normrnd(squeeze(theta(j-1,:,p))',stdw(:,p)); %proposed estimate(s) for one new parameter p (j)
		thetaProp(:,p)=normrnd(thetaCurr(:,p),stdw(:,p)); %proposed estimate(s) for one new parameter p (j)
		%pProp=t2p(thetaProp);
		
		%alpha=alphaFun(thetaProp,thetaCurr,thetaPrev,squeeze(mu(j,:)),squeeze(sigma(j,:,:)), sigmaDet, sigmaInv);
		
		%if p==1
		alpha=...
			alphaLikelihoodFun(thetaProp,thetaCurr) .*...
			alphaPriorHierarchFun(thetaProp,thetaCurr,squeeze(mu(j,:)),squeeze(sigma(j,:,:)),sigmaDet,sigmaInv) .*...
			alphaPriorSpatialFun(thetaProp(:,p),thetaCurr(:,p),thetaCurr(:,p),...
			abs(thetaCurr(:,p)));
			%abs(sqrt(squeeze(sigma(j,p,p))).*thetaCurr(:,p).*SNR));%.* abs(mean(squeeze( iqr(theta(1:j-1,:,p),1)./median(theta(1:j-1,:,p),1) ),2)) );
		%else
		%	alpha=...
		%    alphaLikelihoodFun(thetaProp,thetaCurr) .*...
		%    alphaPriorHierarchFun(thetaProp,thetaCurr,squeeze(mu(j,:)),squeeze(sigma(j,:,:)),sigmaDet,sigmaInv);
		%end
		
		alpha(isnan(alpha))=0; alpha(isinf(alpha))=0; %force data rejection if proposal is not defined
		alpha=min(1,alpha);
		indicatorUpdate=abs( rand(n,1) < alpha );
		indicatorSum(:,p)=indicatorSum(:,p)+indicatorUpdate;
		
		theta(j,:,p)=thetaCurr(:,p).*(1-indicatorUpdate) + indicatorUpdate.*thetaProp(:,p); %update parameter p if accepted proposal
		thetaCurr(:,p)=theta(j,:,p); %update temporary variable thetaCurr (gets reset for every j, but updated within p-loop)
		thetaProp(:,p)=thetaCurr(:,p); %update also proposal!
	end
	
	if j<nburnin
		if mod(j,adaptStepsize)==0
			for p=1:3
				stdw(:,p)=stdw(:,p).*(1-0.234).*( (adaptStepsize+1)./(adaptStepsize+1-indicatorSum(:,p)) );
			end
			indicatorSum=zeros(n,3);
		end
	else
		if mod(j,adaptStepsize)==0
			indicatorSum=zeros(n,3); %only for display
		end
	end
end

end

%% Data preparation
function [bval,data]=prepare_data(bval,data)
[bval,idx_bval]=sort(bval,'ascend'); %sort bval
bval=double(bval);
data=double(data);
if iscolumn(bval)
	bval=bval';
end
if isvector(data)
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

function [Rhat, meanOverall, stdOverall, medianOverall] = EPSR(samples)
%from matjags
% function [R, m, s] = EPSR(samples)
% "estimated potential scale reduction" statistics due to Gelman and Rubin.
% samples(i,j) for sample i, chain j
%
% R = measure of scale reduction - value below 1.1 means converged:
%                                  see Gelman p297
% m = mean(samples)
% s = std(samples)

% This is the same as the netlab function convcalc(samples')

[n,m] = size(samples);
meanPerChain = mean(samples,1); % each column of samples is a chain
meanOverall = mean(meanPerChain);

% Rhat only works if more than one chain is specified.
if m > 1
	% between sequence variace
	B = (n/(m-1))*sum( (meanPerChain-meanOverall).^2);
	
	% within sequence variance
	varPerChain = var(samples);
	W = (1/m)*sum(varPerChain);
	
	vhat = ((n-1)/n)*W + (1/n)*B;
	Rhat = sqrt(vhat/(W+eps));
else
	Rhat = nan;
end

%m = meanOverall;
stdOverall = std(samples(:));
medianOverall=median(samples(:));

%%%%%%%%%
end

function wSum=wSumNeighb(theta,thetaNeighb,weights,mask,idxMaskBoxX,idxMaskBoxY,blockAddSize)
%Input
%theta: theta current or theta proposal; for Metropolis Hastings
%thetaPrev: previous theta (does not work with current theta!)
%mask: nx-ny, should be numerical (no conversion then)
%weights: n-np
%blockAddSize is the additional size from the center: e.g. for 3x3
%neighborhood blockAddSize=1;

%Construct matrices
np=size(theta,2);
thetaImages=zeros([size(mask),np]);
thetaImages(repmat(mask,[1,1,np]))=theta(:);
thetaNeighbImages=zeros([size(mask),np]);
thetaNeighbImages(repmat(mask,[1,1,np]))=thetaNeighb(:);

%Caclulate weighted sum
wSum=zeros([size(mask),np]);
%for p=1:3
for i=idxMaskBoxX(1):idxMaskBoxX(2)
	for j=idxMaskBoxY(1):idxMaskBoxY(2)
		wSum(i,j,:)=...
			sum(sum( abs( thetaNeighbImages(i-blockAddSize:i+blockAddSize,j-blockAddSize:j+blockAddSize,:)-thetaImages(i,j,:) ) ...
			.*mask(i-blockAddSize:i+blockAddSize,j-blockAddSize:j+blockAddSize) ,1),2);
	end
end
wSum(~repmat(mask,[1,1,np]))=0; %only if center pixel is within mask -> cut-off anything outside
wSum=wSum(mask); %vectorize
wSum=wSum./weights; %weighting
end
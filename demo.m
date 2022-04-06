%% Choose scenario
scenario='cancer'; %'stroke' %'in-vivo'

%% Parameters
%for simulation only
if ~strcmp(scenario,'in-vivo')
    snr=30; %[10:10:50,75:25:200];
    bval=[0,10,20,40,80,110,140,170,200,300,400,500,600,700,800,900];
end
%
%LSQ/Bayes fits
pars.nburnin=[1000,5000,2000]; %BFP (LSQ pendant) all, BSP all, MRF all
%pars.ndiscard=0;
pars.nsamples=[1000,5000,2000]; %BFP (LSQ pendant) all, BSP all, MRF all
%pars.adaptStepsize=50;
pars.nchains=4;
pars.doparallel=true;
%pars.overdisp=0;
%spatial prior
pars.spatialPriorFlag=false;
pars.blockAddSize=2; %neighborhood size is 2*blockAddSize + 1 (e.g. 2*1 + 1 = 3 -> 3x3 neighb.)
pars.tauSpatial=1; %1/2 oversmoothes in middle
%hierarchical prior
pars.priorChoice='uniform'; %'uniform' 'normal+jp' 'normal'
pars.paramTransform='no'; %'log' 'no
%likelihood segmentation
pars.segmFlag=false;
pars.bsplit=[200,800];  %1:LSQ/BFP, 2:BSP segm
%start value and constraints for uniform prior (specifically)
pars.pL=[0,0,0]; %only for uniform prior
pars.pU=[2.5e-3,1,50e-3]; %only for uniform prior
%IVIMNET 
parsIVIMNET.reps=1; %repetitions for IVIMNET
parsIVIMNET.batch_size=128;
parsIVIMNET.iter_max=1000;

%% Generate/get data
switch scenario
    case 'cancer' %in-silico
         [data,mask,paramsRef]=ivim_brain_cancer_generate_data(bval,snr);
    case 'stroke' %in-silico
        [data,mask,paramsRef]=ivim_brain_stroke_generate_data(bval,snr);
    case 'in-vivo'
        %For in-vivo data it is assumed that it is already loaded into the
        %workspace as "data"; pre-processing is then performed as in the
        %publication
        %Data size is: nx ny 1 nb ng, where ng is the number of gradient
        %directions
        try
            [data,mask]=ivim_brain_prepare_invivo_data(data);
        catch
            error('Pre-processing of in-vivo data failed: check that pTVreg is added to Matlab path.');
        end
end
[nx,ny,nb]=size(data); %nb=numel(bval);

%% Prepare paths & functions
addpath(genpath('code/'));
addpath(genpath('data/'));
% pathOutput='output/'; %output folder
% mkdir(pathOutput);

%% Prepare Python: for IVIMNET
try
    %ensure that Matlab can find Python
    pe=pyenv;
    if strcmp(pe.Version,"")
        disp('Python not found - please choose python folder');
        folderPython=uigetdir(matlabroot,'Python not found - please choose python folder');
        pyversion(strcat(folderPython,'/python.exe')); %load chosen Python version
    else
        %Python was found
    end
    %install required Python packages:  via command window (not Python terminal)
    system('pip install numpy');
    system('pip install matplotlib');
    system('pip install torch');  %=Pytorch
    system('pip install tqdm');
    %Python successfully set-up
    pythonFlag=true;
    %reload Python script: only necessary after editing
    %clear classes; m = py.importlib.import_module('IVIMNET'); py.importlib.reload(m);
catch
    pythonFlag=false;
    warning('Could not set up Python via Matlab - IVIMNET fits will not be performed. Check also compatibility of Python and Matlab versions: https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf');
end

%% Prepare plots
%parameters to plot
nfit=7;
params=zeros(nx,ny*nfit,3);
%plot
paramStr={'D','f','D*'};
paramsCLims=cat(1,[0,2.5e-3],[0,0.25],[0,15e-3]);
for p=1:3
    hFig{p}=figure('Name',paramStr{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
%time
timeAll=zeros(nfit,1);

%% 1. LSQ (non-linear) non-segmented fit
ifit=1;
%fit parameters
pars_lsq_nonsegm=struct;
pars_lsq_nonsegm.Upper.D=pars.pU(1);
pars_lsq_nonsegm.Upper.f=pars.pU(2);
pars_lsq_nonsegm.Upper.DStar=pars.pU(3);
pars_lsq_nonsegm.Lower.D=pars.pL(1);
pars_lsq_nonsegm.Lower.f=pars.pL(2);
pars_lsq_nonsegm.Lower.DStar=pars.pL(3);
%fit
tic;
[params_lsq_nonlin_nonsegm,paramsVec_lsq_nonlin_nonsegm]=...
    ivim_lsq_nonlin_nonsegm(bval,squeeze(mean(data,5)),mask,pars_lsq_nonsegm);
t_lsq_nonsegm=toc;
%parameters to plot
params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_lsq_nonlin_nonsegm(:,:,1:3);
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=t_lsq_nonsegm;

%% 2. LSQ (non-linear) segmented fit
ifit=2;
%fit parameters
pars_lsq_segm=struct;
pars_lsq_segm.bvalSplit=pars.bsplit(1);
pars_lsq_segm.Upper.D=pars.pU(1);
pars_lsq_segm.Upper.f=pars.pU(2);
pars_lsq_segm.Upper.DStar=pars.pU(3);
pars_lsq_segm.Lower.D=pars.pL(1);
pars_lsq_segm.Lower.f=pars.pL(2);
pars_lsq_segm.Lower.DStar=pars.pL(3);
%fit
tic;
[params_lsq_nonlin_segm,paramsVec_lsq_segm]=...
    ivim_lsq_nonlin_segm(bval,squeeze(mean(data,5)),mask,pars_lsq_segm);
t_lsq_segm=toc;
%parameters to plot
params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_lsq_nonlin_segm(:,:,1:3);
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=t_lsq_segm;

%% 3. IVIMNET
%calling Python from Matlab - works only if everything is set-up
%correctly...
ifit=3;
if pythonFlag
    try
        %fit
        N=sum(mask(:));
        [nx,ny,nz,nb,ng]=size(data);
        dataVec=reshape(data(repmat(mask,[1,1,1,nb,ng])),[N,nb,ng]);
        clear  DStar_IVIMNET_reps D_IVIMNET_reps f_IVIMNET_reps loss_IVIMNET;
        for irep=1:parsIVIMNET.reps
            tic;
            tempOut=py.ivimnet.run(py.numpy.asarray(mean(dataVec./dataVec(:,1,:),3)),...
                uint64(parsIVIMNET.batch_size),uint64(parsIVIMNET.iter_max)); %output order different, see below
            tempOut=cell(tempOut);
            DStar_IVIMNET_reps(:,:,irep)=vec2im(double(tempOut{1}),mask);
            D_IVIMNET_reps(:,:,irep)=vec2im(double(tempOut{2}),mask);
            f_IVIMNET_reps(:,:,irep)=vec2im(double(tempOut{3}),mask);
            loss_IVIMNET_reps(:,irep)=double(tempOut{4});
            t_IVIMNET_reps(irep)=toc;
            
        end
        params_IVIMNET_reps=cat(4,D_IVIMNET_reps,f_IVIMNET_reps,DStar_IVIMNET_reps);
        params_IVIMNET_reps=permute(params_IVIMNET_reps,[1,2,4,3]);
        params_IVIMNET=median(params_IVIMNET_reps,4);
        %parameters to plot
        params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_IVIMNET(:,:,1:3);
        %plot
        for p=1:3
            figure(hFig{p});
            imagesc(params(:,:,p),paramsCLims(p,:)); colorbar;
            axis off; daspect([1,1,1]);
        end
        timeAll(ifit)=sum(t_IVIMNET_reps);
    catch
        disp('Skipping Python-based IVIMNET');
    end
else
    disp('Skipping Python-based IVIMNET');
end
%

%% 4. BSP nonsegm
ifit=4;
%params
pars_bsp_nonsegm=struct;
pars_bsp_nonsegm.nburnin=pars.nburnin(2);
pars_bsp_nonsegm.nsamples=pars.nsamples(2);
pars_bsp_nonsegm.priorChoice='normal+jp';
pars_bsp_nonsegm.paramTransform='log';
pars_bsp_nonsegm.segmFlag=false;
pars_bsp_nonsegm.nchains=pars.nchains;
pars_bsp_nonsegm.doparallel=pars.doparallel;
%fit
tic
[params_bsp_nonsegm,theta_bsp_nonsegm,mu_bsp_nonsegm,sigma_bsp_nonsegm]=...
    ivim_bayes(bval,squeeze(mean(data,5)),mask,pars_bsp_nonsegm);
t_bsp_nonsegm=toc;
%parameters to plot
params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_bsp_nonsegm.median(:,:,1:3);
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=t_bsp_nonsegm;

%% 5. BSP segmented
ifit=5;
%params
pars_bsp_segm=struct;
pars_bsp_segm.nburnin=pars.nburnin(2);
pars_bsp_segm.nsamples=pars.nsamples(2);
pars_bsp_segm.priorChoice='normal+jp';
pars_bsp_segm.paramTransform='log'; %'log' 'no
pars_bsp_segm.segmFlag=true;
pars_bsp_segm.bsplit=pars.bsplit(2);
pars_bsp_segm.nchains=pars.nchains;
pars_bsp_segm.doparallel=pars.doparallel;
%fit
tic
[params_bsp_segm,theta_bsp_segm,mu_bsp_segm,sigma_bsp_segm]=...
    ivim_bayes(bval,squeeze(mean(data,5)),mask,pars_bsp_segm);
t_bsp_segm=toc;
%parameters to plot
params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_bsp_segm.median(:,:,1:3);
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=t_bsp_segm;

%% 6. BSP & MRF nonsegm
%needs values from 5. BSP segmented for prior and for start values
ifit=6;
%params
pars_mrf=struct;
pars_mrf.nburnin=pars.nburnin(3);
pars_mrf.nsamples=pars.nsamples(3);
pars_mrf.nchains=pars.nchains;
pars_mrf.doparallel=pars.doparallel;
pars_mrf.spatialPriorFlag=true;
pars_mrf.blockAddSize=pars.blockAddSize;
pars_mrf.tauSpatial=pars.tauSpatial;
pars_mrf.priorChoice='normal';
pars_mrf.paramTransform='log'; %
%load(strcat(pathOutput,'params_bsp_segm.mat'),'theta_bsp_segm','params_bsp_segm','mu_bsp_segm','sigma_bsp_segm');
pars_mrf.pStart=cat(2,reshape(params_bsp_segm.median(repmat(mask,[1,1,3])),[],3)); %otherwise, it might get stuck!
pars_mrf.muFix=mu_bsp_segm.median;%mean(theta_bsp_segm.median,1);
pars_mrf.sigmaFix=sigma_bsp_segm.median;%cov(theta_bsp_segm.median);
pars_mrf.segmFlag=false;
%fit
tic
[params_mrf,theta_mrf,mu_mrf,sigma_mrf]=...
    ivim_bayes(bval,squeeze(mean(data,5)),mask,pars_mrf);
t_mrf=toc;
%parameters to plot
params(:,(ny*(ifit-1))+1:ny*ifit,:)=params_mrf.median(:,:,1:3);
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=t_mrf;


%% 7. Reference parameters
ifit=7;
%parameters to plot
paramsRef2Plot=paramsRef;
paramsRef2Plot(~repmat(mask,[1,1,3]))=0;
params(:,(ny*(ifit-1))+1:ny*ifit,:)=paramsRef2Plot;
%plot
for p=1:3
    figure(hFig{p});
    imagesc(params(:,:,p),paramsCLims(p,:)); 
    colormap jet; colorbar;
    axis off; daspect([1,1,1]);
end
timeAll(ifit)=0;

%% Note on uncertainties for LSQ methods
%The Bayesian inference function ivim_bayes can also be used to infer
%uncertainties for the LSQ methods (segmented and non-sgemented) with the
%corresponding parameter settings:
%pars.nburnin(1); %short burn-in is suffcient
%pars.nsamples=pars.nsamples(1); %few samples are sufficient
%pars.priorChoice='uniform'; % the LSQ box-constraints correspond to a uniform /flat prior
%pars.paramTransform='no'; % no parameter transformation needed for prior
%pars.segmFlag=false; %false/true for non-segm./segmented fit
%pars.pL=pars.pL; %lower box-constraints
%pars.pU=pars.pU; %upper box-constraints


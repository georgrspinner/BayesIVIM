%IVIM least-squares fit: non-linear (uses Matlab's lsqcurvefit function), segmented
%(first diffusion-only fit, second IVIM model)

%Georg Spinner, Cardio-MR group, ETH & University of Zurich, 22.01.2021

%data: x-y-b
%bval: 1-nb
%mask: x-y
%pars: struct (not required, defaults available)

function [params,paramsVec,...
    rseHigh,residualHigh,exitflagHigh,outputHigh,...
    residualAll,exitflagAll,outputAll]=ivim_lsq_nonlin_segm(bval,data,mask,pars)

%% Default fit parameters
if nargin<4; pars=struct; end; if isempty(pars); pars=struct; end
if nargin<3
    mask=true(size(data,1),size(data,2)); 
else
    if isempty(mask)
        mask=true(size(data,1),size(data,2)); 
    end
end
%---General
%Split b-value: bval>bvalSplit are used for diffusion fit
bvalSplit=get_options(pars,'bvalSplit',300); %can be overwritte, see below
bvalSplit=min(bvalSplit,max(bval(bval<max(bval(:)))));%Make sure there are at least 2 different b-values for D 
%Display warnings
warningDisp=get_options(pars,'warningDisp',false);
if ~warningDisp; warning('off','all'); end
%Debug
pars.debug=get_options(pars,'debug',false);
%---Non-linear
if ~isfield(pars,'Upper'); pars.Upper=struct; end
if ~isfield(pars,'StartPoint'); pars.StartPoint=struct; end
if ~isfield(pars,'Lower'); pars.Lower=struct; end
%Diff: high b-value
pars.Upper.S0=get_options(pars.Upper,'S0',max(abs(data(:))));
pars.StartPoint.S0=get_options(pars.StartPoint,'S0',median(abs(data(:))));
pars.Lower.S0=get_options(pars.Lower,'S0',0);
pars.Upper.S_int=get_options(pars.Upper,'S_int',max(abs(data(:)))); %same default as for S0
pars.StartPoint.S_int=get_options(pars.StartPoint,'S_int',median(abs(data(:)))); %same default as for S0
pars.Lower.S_int=get_options(pars.Lower,'S_int',0); %same default as for S0
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
pars.DStarLargerD=get_options(pars,'DStarLargerD',false); %true: enforces D*>D

%% Prepare data
[bval,data]=prepare_data(bval,data); %sorting, formatting etc.
[nx,ny,nb]=size(data); %size of original data; 
n=sum(mask(:));
if ~isfield(pars,'mask')
    pars.mask=true(nx,ny);
end
%Data to fit: vectorize
dataVec=zeros(n,nb);
for b=1:nb
    temp=data(:,:,b);
    dataVec(:,b)=abs(double(temp(mask)));
end
dataVec(isinf(dataVec))=0; dataVec(isnan(dataVec))=0; 
params=zeros(size(data,1),size(data,2),5); paramsVec=zeros(n,5); %D,f,D*,S0,S_int

%% High b-values: D & S_int
hWaitHigh = waitbar(0,'IVIM LSQ: high b-values => D, S_{int}'); 
optionsHigh = optimoptions('lsqcurvefit','Jacobian','on','Display','off');
%optionsHigh = optimoptions('lsqcurvefit','PlotFcn',@optimplotresnorm); %Debug
xHigh=zeros(n,2); rseHigh=zeros(n,1); residualHigh=zeros(n,numel(bval(bval>=bvalSplit))); exitflagHigh=zeros(n,1); outputHigh=cell(n,1);
if pars.debug
    hFigDebugHigh=figure;
    bvalPlotDebug = linspace(min(bval(bval>=bvalSplit)),bval(end));
end
for i=1:n
    try
    waitbar(i/n,hWaitHigh,['IVIM LSQ: high b-values => D, S_{int} ',num2str(round(i/n*100)),'%']);
    catch
    end
    [xHigh(i,:),rseHigh(i),residualHigh(i,:),exitflagHigh(i),outputHigh{i}] =...
        lsqcurvefit(@(x,xdata) funHigh(x,xdata),[pars.StartPoint.D,pars.StartPoint.S_int],bval(bval>=bvalSplit),dataVec(i,bval>=bvalSplit),...
        [pars.Lower.D,pars.Lower.S_int],[pars.Upper.D,pars.Upper.S_int],optionsHigh);
    if pars.debug
        figure(hFigDebugHigh);
        semilogy(bval(bval>=bvalSplit),dataVec(i,bval>=bvalSplit)/dataVec(i,1),'ro',bvalPlotDebug,funHigh(xHigh(i,:),bvalPlotDebug)/dataVec(i,1),'b-')
        axis([bval(1),bval(end),0.2,1.2]);
        %legend('Data','Fitted mono-exponential')
        title('Data and Fitted Curve')
    end
end
if pars.debug
    close(hFigDebugHigh);
end
%D
paramsVec(:,1)=xHigh(:,1); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,1); params(:,:,1)=temp;
%S_int
paramsVec(:,5)=xHigh(:,2); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,5); params(:,:,5)=temp;
try
close(hWaitHigh);
catch
end

%% All b-values: f & D*
hWaitAll = waitbar(0,'IVIM LSQ: all b-values => f, D*'); 
optionsAll = optimoptions('lsqcurvefit','Jacobian','on','Display','off');
%optionsHigh = optimoptions('lsqcurvefit','PlotFcn',@optimplotresnorm); %Debug
xAll=zeros(n,2); rseAll=zeros(n,1); residualAll=zeros(n,numel(bval)); exitflagAll=zeros(n,1); outputAll=cell(n,1);
if pars.DStarLargerD %voxel-wise constraints for D*
    DStarLower=max(repmat(pars.Lower.DStar,[n,1]),xHigh(:,1)); %voxel-wise DStar lower bound if D*>D is enforced
else
    DStarLower=repmat(pars.Lower.DStar,[n,1]); %replicate box constraint
end
if pars.debug
    hFigDebugAll=figure;
    bvalPlotDebug = linspace(min(bval),bval(end));
end
for i=1:n
%     if sum( ((xHigh(i,2)*exp(-xHigh(i,1)*bval(bval<bvalSplit)))-dataVec(i,bval<bvalSplit))<0 )<sum(bval<bvalSplit)
%     else
%         continue
%     end
    try
    waitbar(i/n,hWaitAll,['IVIM LSQ: all b-values => f, D* ',num2str(round(i/n*100)),'%']);
    catch
    end
    [xAll(i,:),rseAll(i),residualAll(i,:),exitflagAll(i),outputAll{i}] =...
        lsqcurvefit(@(x,xdata) funAll(x,xdata,xHigh(i,1),xHigh(i,2)),[pars.StartPoint.f,pars.StartPoint.DStar],bval,...
        dataVec(i,:),... %alternative: S(b)/S_int - exp(-b*D)
        [pars.Lower.f,DStarLower(i)],[pars.Upper.f,pars.Upper.DStar],optionsAll);
    if pars.debug
        figure(hFigDebugAll)
        semilogy(bval,dataVec(i,:)/dataVec(i,1),'ro',...
            bvalPlotDebug,funAll(xAll(i,:),bvalPlotDebug,xHigh(i,1),xHigh(i,2))/dataVec(i,1),'b-',...
            bvalPlotDebug,funHigh(xHigh(i,:),bvalPlotDebug)/dataVec(i,1),'b--');
        %axis([min(0,0.9*bval(1)),1.1*bval(end),0.25,1.2]);
        axis([bval(1),bval(end),0.2,1.2]);
        %legend('Data','Fitted bi-exponential')
        title(['Data and Fitted Curve','-',num2str(i),'/',num2str(n)]);
    end
end
if pars.debug
    close(hFigDebugAll);
end
paramsVec(:,2)=xAll(:,1); %f
temp=zeros(nx,ny); temp(mask)=paramsVec(:,2); params(:,:,2)=temp;
paramsVec(:,3)=xAll(:,2); %DStar
temp=zeros(nx,ny); temp(mask)=paramsVec(:,3); params(:,:,3)=temp;
paramsVec(:,4)=xHigh(:,2)./(1-xAll(:,1)); %S_int/(1-f)
temp=zeros(nx,ny); temp(mask)=paramsVec(:,4); params(:,:,4)=temp;
try
close(hWaitAll);
catch
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


function [fOut,grad]=funHigh(x,xdata)
b=xdata;
D=x(:,1);
S_int=x(:,2);
fun=@(b,D,S_int) S_int*exp(-b*D);
fOut=fun(b,D,S_int);
if nargout>1 %gradient(fun(b,S_int,D))
    grad1=@(b,D,S_int) (-b).*S_int.*exp(-D*b); %diff. w.r.t. D
    grad(1,:)=grad1(b,D,S_int);
    grad2=@(b,D,S_int) exp(-D*b); %diff. w.r.t. S_int
    grad(2,:)=grad2(b,D,S_int);
    grad=grad';
end
end

function [fOut,grad]=funAll(x,xdata,D,S_int) %
b=xdata;
f=x(:,1);
DStar=x(:,2);
fun=@(b,f,DStar) S_int*( ((f/(1-f))*exp(-DStar*b)) + exp(-D*b) );
fOut=fun(b,f,DStar);
if nargout>1 %gradient(fun(b,D,S_int,f,DStar))
    grad1=@(b,f,DStar) S_int*(((1-f)^(-2)) * exp(-DStar*b)); %diff. w.r.t. f
    grad(1,:)=grad1(b,f,DStar);
    grad2=@(b,f,DStar) S_int*((f/(1-f))*exp(-DStar*b).*(-b)); %diff. w.r.t. D*
    grad(2,:)=grad2(b,f,DStar);
    grad=grad';
end

end
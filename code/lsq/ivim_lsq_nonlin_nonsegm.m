%IVIM least-squares fit: non-linear (uses Matlab's lsqcurvefit function), non-segmented
%(IVIM model for all b-values)

%Georg Spinner, Cardio-MR group, ETH & University of Zurich, 22.01.2021

%data: x-y-b
%bval: 1-nb
%mask: x-y
%pars: struct not required, defaults available)

function [params,paramsVec,...
    rse,residual,exitflag,output]=ivim_lsq_nonlin_nonsegm(bval,data,mask,pars)

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
params=zeros(size(data,1),size(data,2),4); paramsVec=zeros(n,4); %D,f,D*,S0,S_int

%% All b-values: D, f, D* & S0
hWait = waitbar(0,'IVIM LSQ: non-segmented'); 
options = optimoptions('lsqcurvefit','Jacobian','on','Display','off');
%options = optimoptions('lsqcurvefit','PlotFcn',@optimplotresnorm); %Debug
x=zeros(n,4); rse=zeros(n,1); residual=zeros(n,numel(bval)); exitflag=zeros(n,1); output=cell(n,1);
if pars.debug
    hFigDebug=figure;
    bvalPlotDebug = linspace(min(bval),bval(end));
end
funHigh=@(x,xdata) x(:,4)*(1-x(:,2))*exp(-x(:,1)*xdata); %S0*(1-f)*exp(-bD)
for i=1:n
    try
    waitbar(i/n,hWait,['IVIM LSQ: non-segmented - ',num2str(round(i/n*100)),'%']);
    catch
    end
    [x(i,:),rse(i),residual(i,:),exitflag(i),output{i}] =...
        lsqcurvefit(@(x,xdata) fun(x,xdata),...
        [pars.StartPoint.D,pars.StartPoint.f,pars.StartPoint.DStar,pars.StartPoint.S0],...
        bval,dataVec(i,:),...
        [pars.Lower.D,pars.Lower.f,pars.Lower.DStar,pars.Lower.S0],...
        [pars.Upper.D,pars.Upper.f,pars.Upper.DStar,pars.Upper.S0],...
        options);
    if pars.debug
        figure(hFigDebug);
        semilogy(bval,dataVec(i,:)/dataVec(i,1),'ro',...
            bvalPlotDebug,funOpt(x(i,:),bvalPlotDebug)/dataVec(i,1),'b-',...
            bvalPlotDebug,funHigh(x(i,:),bvalPlotDebug)/dataVec(i,1),'b--');
        axis([bval(1),bval(end),0.2,1.2]);
        %legend('Data','Fitted mono-exponential')
        title('Data and Fitted Curve')
    end
end
if pars.debug
    close(hFigDebug);
end
%D
paramsVec(:,1)=x(:,1); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,1); params(:,:,1)=temp;
%f
paramsVec(:,2)=x(:,2); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,2); params(:,:,2)=temp;
%DStar
paramsVec(:,3)=x(:,3); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,3); params(:,:,3)=temp;
%S_int
paramsVec(:,4)=x(:,4); 
temp=zeros(nx,ny); temp(mask)=paramsVec(:,4); params(:,:,4)=temp;
try
close(hWait);
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


function [fOut,grad]=fun(x,xdata)
b=xdata;
D=x(:,1);
f=x(:,2);
DStar=x(:,3);
S0=x(:,4);
fun=@(b,D,f,DStar,S0) S0*( f*exp(-b*DStar) + (1-f)*exp(-b*D) );
fOut=fun(b,D,f,DStar,S0);
if nargout>1 %gradient(fun(b,S_int,D))
    grad1=@(b,D,f,DStar,S0)  S0*( (1-f)*exp(-b*D).*(-b) ); %diff. w.r.t. D
    grad(1,:)=grad1(b,D,f,DStar,S0);
    grad2=@(b,D,f,DStar,S0)  S0*( exp(-b*DStar) + (-1)*exp(-b*D) ); %diff. w.r.t. f
    grad(2,:)=grad2(b,D,f,DStar,S0);
    grad3=@(b,D,f,DStar,S0)  S0*( f*exp(-b*DStar).*(-b) ); %diff. w.r.t. D*
    grad(3,:)=grad3(b,D,f,DStar,S0);
    grad4=@(b,D,f,DStar,S0)   f*exp(-b*DStar) + (1-f)*exp(-b*D); %diff. w.r.t. S0
    grad(4,:)=grad4(b,D,f,DStar,S0);
    grad=grad';
end
end



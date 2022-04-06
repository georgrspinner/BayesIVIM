function [data,DRef,fRef,DStarRef,datasos,data_noiseless_cc,dataC]=ivim_brain_simulate(rec,sens,im,bval,d,f,ds,snr)

%% Prepare
[nx,ny,nc]=size(rec);
nb=numel(bval);

%% Get masks
%ordering in ascending magnitude order:
%cancer: bone & background, WM, skin, cancer: necrotic, GM, CSF pv, cancer: surrounding, cancer: center, CSF pure
%stroke: bone & background, WM, skin, GM, 1 overlap near CSF pv, CSF pv, infarc core+overlap, infarct surrounding, CSF pure
magn_uni=unique(im(:));
nuni=numel(magn_uni);
mask_uni=false(nx,ny,nuni);
for iu=1:nuni
	mask_uni(:,:,iu)=im==magn_uni(iu);
end
%mask_uni=mask_uni & repmat(support,[1,1,nuni]);

%% Simulate data
%Parameter maps: reference
DRef=sum(bsxfun(@times,mask_uni,permute(d,[1,3,2])),3);
fRef=sum(bsxfun(@times,mask_uni,permute(f,[1,3,2])),3);
DStarRef=sum(bsxfun(@times,mask_uni,permute(ds,[1,3,2])),3);
%IVIM model
ivim=@(b,d,f,ds) f.*exp(-b*ds) + (1-f).*exp(-b*d);
ivim_factor=zeros(nx,ny,nb);
for ib=1:numel(bval)
    ivim_factor(:,:,ib)=ivim(bval(ib),DRef,fRef,DStarRef);
end
data_noiseless=repmat(rec,[1,1,1,nb]).*permute(repmat(ivim_factor,[1,1,1,nc]),[1,2,4,3]);
%Correct SNR
%data_noiseless=data_noiseless/0.22; %normalize magnitude to 1 in WM (SNR is then relative to WM-SNR)
data_noiseless_cc=sum(data_noiseless.*conj(sens),3)./sum(conj(sens).*sens,3);
mask=logical(sum(mask_uni(:,:,2:end),3));
mean_magn=mean(abs(data_noiseless_cc(mask)));
%Add noise
%dataC=data_noiseless + (1/snr)*complex(randn(nx,ny,nc,nb),randn(nx,ny,nc,nb));
%dataC=data_noiseless + (mean_magn/snr)*complex(randn(nx,ny,nc,nb),randn(nx,ny,nc,nb));
dataC=data_noiseless  + (1/snr)*complex(randn(nx,ny,nc,nb),randn(nx,ny,nc,nb));
data=zeros(nx,ny,1,nb); %nz=1
datasos=zeros(nx,ny,1,nb);
for ib=1:numel(bval)
    data(:,:,:,ib)=sum(dataC(:,:,:,ib).*conj(sens),3)./sum(conj(sens).*sens,3); %Roemer combination
    datasos(:,:,:,ib)=sqrt(sum(abs(dataC(:,:,:,ib)).^2,3));
end
data=abs(data); %magnitude (from complex data)

end
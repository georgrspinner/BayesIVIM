function [data,mask,paramsRef]=ivim_brain_cancer_generate_data(bval,snr)
%parameters
coilAmpFactor=2.55;
d= [0, 0.6,  0, 0.4,  0.8,   1.25,  1.2,   1.4 ,  2.5]*1e-3;
f= [0, 0.05, 0, 0.01, 0.08,  0,     0.10,  0.15,  0];
ds=[0, 4,    0, 0.1,  6,     0,     10,     12,   0]*1e-3;
%1bone & background, 2WM, 3skin, 4cancer: necrotic, 5GM, 6CSF pv, 7cancer: surrounding, 8cancer: center, 9CSF pure

%simulate
load('data/cancer/BrainCancer.mat','rec','im','sensitivity');
[data,DRef,fRef,DStarRef,datasos,data_noiseless_cc,dataC]=ivim_brain_simulate(rec,sensitivity,im,bval,d,f,ds,snr*coilAmpFactor/0.22); %given SNR (without factor) would be only correct if coil sensitivity were spatially homogeneous
paramsRef=cat(3,DRef,fRef,DStarRef);

%segmentation: mask
load('data/mask_brain_sim.mat','mask');
mask(DRef==2.5e-3 | DRef==1.25e-3)=false; %hard-coded: remove all CSF parts
paramsRef(~repmat(mask,[1,1,3]))=0;

end
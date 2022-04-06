function [dataReg,mask]=ivim_brain_prepare_invivo_data(data)

%parameters: registration
parsReg=struct;
parsReg.pix_resolution = [1,1];
parsReg.metric = 'nuclear';
parsReg.grid_spacing = [4, 4];
parsReg.isoTV = 3e-2;
parsReg.jac_reg = 2;
parsReg.spline_order = 1;
parsReg.display = 'off';
parsReg.border_mask = 6;
parsReg.k_down = 0.7;
parsReg.max_iters = 120;

%registration: preparation
[nx,ny,nz,nb,ng]=size(data);
imgs = double(reshape(data, [nx, ny, 1, 1, nb*ng]));
imgs = sqrt(imgs); %sqrt is a dirty trick to get rid of highlights
maximgs=max(max(imgs, [], 1), [], 2);
imgs = imgs ./ maximgs;

%registration: calculate
tic
[voldef_pl, Tmin_pl,  Kmin_pl] = ptv_register(imgs, [], parsReg);
toc

%registration: convert data back
dataReg=voldef_pl .* maximgs;
dataReg=dataReg.^2;
dataReg=reshape(dataReg,[nx,ny,1,nb,ng]);


%parameters segmentation
parsSeg.bvalSplit=200;
parsSeg.maxDSeg=1.25e-3;%2.5e-3; %2e-3
parsSeg.maxfSeg=0.25; %0.4

%segmentation
se = strel('disk',1);
[maskSegment,maskedImage,maskCSF] = segmentImageBrainDWI(mean(mean(abs(dataReg),5),4));
params=ivim_lsq_loglin_segm(bval,squeeze(mean(dataReg,5)),maskSegment,parsSeg);
maskD=params.D<parsSeg.maxDSeg;
maskf=params.f<parsSeg.maxfSeg;
mask=maskSegment ;
mask=mask & maskD;
mask=mask & maskf;
mask=imerode(mask,se);
mask=imclose(mask,se);

end
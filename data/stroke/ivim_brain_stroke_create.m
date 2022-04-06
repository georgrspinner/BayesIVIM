%% Requires http://bigwww.epfl.ch/algorithms/mriphantom/, see also https://doi.org/10.1109/TMI.2011.2174158
%Allows re-creation of numerical phantoms

%% Parameters
res=192;

%% PHANTOM

%Design Phantom
%Standard phantom
DefineBrain;

%1. Penumbra: 0.4 (draw whole area)
BrainSurrounding=DesignPhantom(zeros(res,res),Brain);
%2. Ischemic core: 0.2 (follow contours from before, but leave out some
%ares within the area -> ischemic core)
BrainStroke=DesignPhantom(zeros(res,res),BrainSurrounding);
%Combine
Brain_stroke=Brain; %copy standard
Brain_stroke.region{end+1}=BrainSurrounding.region{end}; %add surrounding
Brain_stroke.region{end+1}=BrainStroke.region{end}; %add surrounding
%im = RasterizePhantom(Brain_stroke,res); %check
%Export phantom
ExportPhantom(Brain_stroke,'DefineBrain_stroke');
%Check
%DefineBrain_stroke;
%im = RasterizePhantom(DefineBrainstroke,res);
%Load pahtom
DefineBrain_stroke; %Loads variable "DefineBrainStroke"
BrainStroke=DefineBrainstroke; clear DefineBrainstroke;
Brain=BrainStroke;

%% Coils simulation
coil.Nb_coils = 8;
coil.res = res;
coil.type = 'biot';
coil.param.FS = 0.28; % FOV width
coil.param.D = 0.17; % Distance center->coil
coil.param.R = 0.05; % radius of the coil
coil.param.rand = 0;
coil = simulate_sensitivities(coil);

NbCoils = size(coil.sensitivity,3);

sens.model = 'sinusoidal'; %better than polynomial, especially for beziers...
sens.param = 7; %default for sinusoidal...

im = RasterizePhantom(Brain,res,[1],0);

support = (im>1e-3);numel(find(support));
sensitivity = coil.sensitivity/max(reshape(abs(coil.sensitivity.*repmat(support,[1,1,NbCoils])),1,numel(coil.sensitivity)));
for c=1:NbCoils
	c
	sens_brain(c) = SensFitting(sensitivity(:,:,c),sens.model,sens.param,support);
end

%% K-SPACE
k = GenerateFullCart2DKspace(res*[1,1],Brain.FOV);

%% MR DATA
for c=1:NbCoils
	c
tic;m_brain(c,:) = MRData(Brain,sens_brain(c),k);t_analytical(c)=toc;
end

for c=1:NbCoils
	c
%rec(:,:,c)  = ReconsFromFullCartesianKspace(m_brain(c,:) .* (1 + complex(randn(size(m_brain,2),1),randn(size(m_brain,2),1))'),k,Brain.FOV);
rec(:,:,c)  = ReconsFromFullCartesianKspace(m_brain(c,:),k,Brain.FOV);
end

%% Save
save('BrainStroke.mat');


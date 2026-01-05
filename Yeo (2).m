% addpath /imaging/duncan/users/ma07/Software/fieldtrip-20190724
% addpath /imaging/local/software/freesurfer/6.0.0/x86_64/matlab
% ft_defaults
% addpath('/imaging/local/software/spm_cbu_svn/releases/spm12_latest')
% addpath(genpath('/imaging/duncan/users/ye02/intraop/Analysis/'))

%%

%% set correct directories
% clear; clc;
addpath /home/eng/hartmas/'MATLAB Add-Ons'/Collections/FieldTrip/external/freesurfer/
load('/mnt/jane_data/Intraop-Cam/Analysis/Moataz/elec_localization/templates/elecs_n79_coords_realigned_ft_format_hermes.mat')
templates_dir =  '/mnt/jane_data/Intraop-Cam/Analysis/Moataz/elec_localization/templates';

%%
elecs_ft_realigned_hermes.chanpos=[elecs_ft_realigned_hermes.chanpos; table2array(loc_to_add)];
%%
elecs_ft_realigned_hermes.elecpos=[elecs_ft_realigned_hermes.elecpos; table2array(loc_to_add)];
%%
elecs_ft_realigned_hermes.label(80:83)=cellstr(['80';'81';'82';'83']);
%%
elecs_ft_realigned_hermes.tra=eye(83);
%%
%% load data files

%specify bands to plot
bands = [1 4; 4 8; 8 12; 12 30; 30 70; 70 250];
band = bands(1,:);
reref_scheme = {'reref_bipolar'};
conds             = {'alt' 'countF'};

%% set up params for fieldtrip surface rendering
face_color = 0.8*[1 1 1];
mesh = [];
% [mesh.pos,mesh.tri] = freesurfer_read_surf(fullfile(templates_dir,'MNI_0.8_native_cras_lh.pial'));
[mesh.pos,mesh.tri] = read_surf(fullfile(templates_dir,'MNI_0.8_native_cras_lh.pial'));

%%%% put electrode coords into fieldtrip structure
% elec_realigned_coords = [[elecs.coord_MNI_shift_x]' [elecs.coord_MNI_shift_y]' [elecs.coord_MNI_shift_z]'];

load(fullfile(templates_dir, 'SubjectUCI29_elec_acpc_f.mat'));

%% load elecs coords

nifti_dir = '/home/eng/hartmas/MATLAB Add-Ons/Collections/FieldTrip/template/atlas/yeo/';
nifti_fname = 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_colin27.nii';
atlas = ft_read_atlas(fullfile(nifti_dir,nifti_fname));
atlas.coordsys = 'mni';


volume = ft_checkdata(atlas, 'datatype', 'source');
xyz = [volume.pos]';
mask_inds = volume.tissue > 0;
mask_locs = find(mask_inds == 1);
mask_xyz = xyz(:,mask_inds);
range = 10;


for i = 1:83
    poi = elecs_ft_realigned_hermes.chanpos(i,:);
    dist = pdist2(poi,mask_xyz','euclidean');
    min_dist(i) = min(dist);
    labels_withinrange = volume.tissue(mask_locs(find( (dist>min(dist)) & (dist <(min(dist)+range)) )));
    unique_labels = unique(labels_withinrange);
    labels_freq = histc(labels_withinrange,unique_labels);
    mostfreqlabel = unique_labels(find(labels_freq == max(labels_freq)));
    elec_label(i) = mostfreqlabel;
end



%% Select electrodes
all_elecs = logical(1:length(elec_label));
indep_bip_elecs_2skip = all_elecs; indep_bip_elecs_2skip([2:2:70 71 73:2:83]) = 0;
indep_elecs = all_elecs  & indep_bip_elecs_2skip;
dep_elecs = all_elecs

%% overlap of MD masks with sig electrodes
elecs2use  = dep_elecs'; %%%%%%%%%%%%%% <-------- change
% FPN1_DMN2_MOT3_DAN4_CON5
FPN_elecs_inds = (elec_label == 6)' & elecs2use; 
DMN_elecs_inds = (elec_label == 7)' & elecs2use; 
MOTOR_elecs_inds = (elec_label == 2)' & elecs2use; 
DAN_elecs_inds = (elec_label == 3)' & elecs2use; 
CON_elecs_inds = (elec_label == 4)' & elecs2use; 
unlabelled_elecs_inds = (elec_label == 0)' & elecs2use;

%% plot figure
f1 = figure(1); clf(1);
mesh.tri = mesh.tri+1;
ft_plot_mesh(mesh,'facecolor',face_color);
set(gcf, 'Color', 'w'); % Set the background color of the figure to white

elecs2useindx = find(elecs2use == 1);

for E = 1:length(elecs2useindx)%1:length(elecs)
    
    e = elecs2useindx(E);
    %ft struct for each  post corrected electrode
    subj_elecs_ft_realigned_hermes.elecpos      = elecs_ft_realigned_hermes.elecpos(e,:);
    subj_elecs_ft_realigned_hermes.chanpos     =elecs_ft_realigned_hermes.chanpos(e,:);
    subj_elecs_ft_realigned_hermes.label            = elecs_ft_realigned_hermes.label(e);
    subj_elecs_ft_realigned_hermes.tra              = [1]; %eye(length(each_elecs_ft_realigned_hermes.elecpos ));
    subj_elecs_ft_realigned_hermes.unit    = 'mm';
    
    if FPN_elecs_inds(e) == 1
            elec_color = [230 148 34]./255;
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
            disp(subj_elecs_ft_realigned_hermes)
            disp('here')
    elseif DMN_elecs_inds(e) == 1
            elec_color = 'r';
            %elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    elseif MOTOR_elecs_inds(e) == 1
            elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    elseif DAN_elecs_inds(e) == 1
            elec_color = 'g';
            %elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    elseif CON_elecs_inds(e) == 1
            elec_color = 'm';
            %elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    elseif unlabelled_elecs_inds(e) == 1
            elec_color = 'k';
            %elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    else
            elec_color = 'k';
            %elec_color = 'b';
            ft_plot_sens(subj_elecs_ft_realigned_hermes,'facecolor',elec_color,'elecshape','sphere');
    end
    
end %elecs of each subj

material dull;
lighting gouraud;
view([-100 20])
camlight;

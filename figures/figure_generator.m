% TCGA-HT-7884/

mask = load_untouch_nii('truth.nii.gz');
mask = mask.img;
unique(mask(:));
mask(mask==2) = 1;

flair = load_untouch_nii('flair.nii.gz');
flair = flair.img;

t1 = load_untouch_nii('t1.nii.gz');
t1 = t1.img;

t2 = load_untouch_nii('t2.nii.gz');
t2 = t2.img;

t1post = load_untouch_nii('t1Gd.nii.gz');
t1post = t1post.img;

x = 151; y = 109; z = 65;
view = {'x','y','z'};
modality = {'flair','t1','t2','t1Gd'};

mask_z = squeeze(mask(:,:,z));
mask_y = imrotate(squeeze(mask(:,y,:)),90);
mask_x = imrotate(squeeze(mask(x,:,:)),90);
% 
% flair_z = squeeze(flair(:,:,z));
% flair_y = imrotate(squeeze(flair(:,y,:)),90);
% flair_x = imrotate(squeeze(flair(x,:,:)),90);
% 
% [flair_z, flair_bound_z] = getROIbox(flair_z,flair_z);
% [flair_y, flair_bound_y] = getROIbox(flair_y,flair_y);
% [flair_x, flair_bound_x] = getROIbox(flair_x,flair_x);
% 
% mask_z_flair = mask_z(flair_bound_z(1,1):flair_bound_z(1,2),...
%                       flair_bound_z(2,1):flair_bound_z(2,2));
% 
% [h_mask, h_t2] = pausOverlay(flair_z, mask_z_flair, ...
%     [0 max(flair_z(:))], [0 2], 'hot', 0.3, ...
%     [0:size(flair_z,1)], [0:size(flair_z,2)], '');

fig_path = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/1p_19q/figures';
for i = 1:length(modality)
    
    img = load_untouch_nii([modality{i},'.nii.gz']);
    img = img.img;
    
    for view_ind = 1:length(view)
        if strcmpi(view{view_ind}, 'z')
            
            slice = squeeze(img(:,:,z));
            
            [slice_z, slice_bound_z] = getROIbox(slice,slice);
            
            
            mask_z_slice = mask_z(slice_bound_z(1,1): slice_bound_z(1,2),...
                                slice_bound_z(2,1): slice_bound_z(2,2));
            
            [h_mask, h_slice, figH] = pausOverlay(slice_z, mask_z_slice, ...
                [0 max(slice_z(:))], [0 2], 'hot', 0.3, ...
                [0:size(slice_z,1)], [0:size(slice_z,2)], '');
            
            set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
%             img = frame2im(getframe(gca));
            filename = fullfile(fig_path,[modality{i},'_',view{view_ind},'.tif']);
            saveas(figH,filename);
            export_fig(filename)
            close all
            
        elseif strcmpi(view{view_ind}, 'x')
            
            slice = imrotate(squeeze(img(x,:,:)),90);
            
            [slice_x, slice_bound_x] = getROIbox(slice,slice);

            mask_x_slice = mask_x(slice_bound_x(1,1): slice_bound_x(1,2),...
                                slice_bound_x(2,1): slice_bound_x(2,2));
            
            [h_mask, h_slice, figH] = pausOverlay(slice_x, mask_x_slice, ...
                [0 max(slice_x(:))], [0 2], 'hot', 0.3, ...
                [0:size(slice_x,1)], [0:size(slice_x,2)], '');
            
            set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
            filename = fullfile(fig_path,[modality{i},'_',view{view_ind},'.tif']);
            saveas(figH,filename);
            export_fig( filename)
            close all
            
         elseif strcmpi(view{view_ind}, 'y')
            
            slice = imrotate(squeeze(img(:,y,:)),90);
            
            [slice_y, slice_bound_y] = getROIbox(slice,slice);

            mask_y_slice = mask_y(slice_bound_y(1,1): slice_bound_y(1,2),...
                                slice_bound_y(2,1): slice_bound_y(2,2));
            
            [h_mask, h_slice, figH] = pausOverlay(slice_y, mask_y_slice, ...
                [0 max(slice_y(:))], [0 2], 'hot', 0.3, ...
                [0:size(slice_y,1)], [0:size(slice_y,2)], '');     
            
            set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
            filename = fullfile(fig_path,[modality{i},'_',view{view_ind},'.tif']);
            saveas(figH,filename);
            export_fig(filename)
            close all
            
        end
        
        
    end
    
    
end

% Patch Generation figure

cd(fullfile(fig_path,'patches'))

files = dir;
files(1:2) = [];

for i = 1: length(files)
   
    patch = readNPY(files(i).name);
    
    figH = figure;
    imshow(patch, [])
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

    saveas(figH,[files(i).name(1:end-4),'.tif'])
    export_fig([files(i).name(1:end-4),'.tif'])
    close all
       
    
end


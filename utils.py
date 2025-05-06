import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


#sets the seed
def set_seed(seed):
    np.random.seed(seed)#sets seed for np
    torch.manual_seed(seed) #sets seed for torch
    return

#Generates an impulse input from a location
def impulse_input_from_loc(loc, shape):
    impulse_input = torch.zeros(shape)
    r = loc[1] 
    c = loc[0] #Flip
    impulse_input[r,c] = 1
    return impulse_input


#Gets the model accuracy

def get_test_accuracy_new(model_real, model_imag, loader, criterion):
    #Set to eval
    model_real.eval()
    model_imag.eval()

    running_loss = 0.0
    #Get device
    device = next(model_real.parameters()).device

    for i, data in enumerate(loader, 0):
        inputs, MRIs, masks, labels = data[0][0], data[0][1], data[0][2], data[1]

        with torch.no_grad():
            #Calculate loss
            loc, mri = inputs.to(device), MRIs.to(device) 
            
            outputs_real = model_real(loc, mri)
            outputs_imag = model_imag(loc, mri)
            outputs = outputs_real + 1j*outputs_imag
            
            
            #Mask the border
            masks_new = masks.to(device)        
            outputs = outputs.to(device) * masks_new
            labels = labels.to(device) * masks_new
            
            #Loss
            loss = criterion(outputs, labels) #/ len(dataset) 
            n_batch = loader.batch_size
            norm_factor_batch = len(loader.dataset)/n_batch
            running_loss += loss.item()/norm_factor_batch
                        
    #set back to train
    model_real.train()
    model_imag.train()
    
    #Return
    return running_loss #.cpu()

def get_test_accuracy(model_real, model_imag, dataset, criterion):
    #Set to eval
    model_real.eval()
    model_imag.eval()
    
    #Get device
    device = next(model_real.parameters()).device

    with torch.no_grad():
        #Calculate loss
        loc, mri = dataset.loc.to(device), dataset.transformMRI(dataset.MRI).to(device) 
        
        outputs_real = model_real(loc, mri)
        outputs_imag = model_imag(loc, mri)
        outputs = outputs_real + 1j*outputs_imag
        
        
        #Mask the border
        masks_new = dataset.masks.to(device)        
        outputs = outputs * masks_new
        labels = dataset.transformF(dataset.fdata).to(device) * masks_new
        
        #Loss
        loss = criterion(outputs, labels) #/ len(dataset)
    
    #set back to train
    model_real.train()
    model_imag.train()
    
    #Return
    return loss.cpu()

# applies padding: MRI
def apply_pad(MRI, tot_x_axis_len, tot_y_axis_len):
    v = torch.zeros(tot_y_axis_len, tot_x_axis_len, dtype=MRI.dtype)# + MRI.min()
    dim1_diff = int((v.shape[0]-MRI.shape[0])/2)
    dim2_diff = int((v.shape[1]-MRI.shape[1])/2)
    v[dim1_diff:MRI.shape[0]+dim1_diff, dim2_diff:MRI.shape[1]+dim2_diff] = MRI
    return v

# applies padding: location
def apply_pad_loc(loc, tot_x_axis_len, tot_y_axis_len):
    new_loc = torch.zeros(loc.shape)
    dim1_diff = int((tot_y_axis_len-434)/2)
    dim2_diff = int((tot_x_axis_len-362)/2)
    new_loc[:,0] = loc[:,0]+dim2_diff
    new_loc[:,1] = loc[:,1]+dim1_diff    
    return new_loc

# removes padding: MRI
def remove_pad(v, tot_x_axis_len, tot_y_axis_len):
    dim1_diff = int((v.shape[0]-434)/2)
    dim2_diff = int((v.shape[1]-362)/2)
    MRI = v[dim1_diff:434+dim1_diff, dim2_diff:362+dim2_diff]
    return MRI

# applies padding: location
def remove_pad_loc(loc, tot_x_axis_len, tot_y_axis_len):
    old_loc = torch.zeros(loc.shape)
    dim1_diff = int((tot_y_axis_len-434)/2)
    dim2_diff = int((tot_x_axis_len-362)/2)
    old_loc[:,0] = loc[:,0]-dim2_diff
    old_loc[:,1] = loc[:,1]-dim1_diff    
    return old_loc

def complex_padding(fdata, tot_y_axis_len, tot_x_axis_len):
    x_real_pad = torch.zeros(fdata.shape[0], tot_y_axis_len, tot_x_axis_len)
    x_imag_pad = torch.zeros(fdata.shape[0], tot_y_axis_len, tot_x_axis_len)
    for i in range(fdata.shape[0]):
        x_real = fdata[i,:,:].real.clone()
        x_imag = fdata[i,:,:].imag.clone()
        x_real_pad[i,:,:] = apply_pad(x_real, tot_x_axis_len, tot_y_axis_len)
        x_imag_pad[i,:,:] = apply_pad(x_imag, tot_x_axis_len, tot_y_axis_len)
    y = x_real_pad + 1j*x_imag_pad    
    return y

def pearson_corr_real(x, y):
    x = x.abs()
    y = y.abs()
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearson_corr_real_val = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr_real_val
    
def pearson_corr_real_angle(x, y):
    x = x
    y = y
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearson_corr_real_val = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr_real_val

def pearson_corr_angle_sin(x, y):
    x = x
    y = y
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearson_corr_real_val = torch.sum(torch.sin(vx) * torch.sin(vy)) / (torch.sqrt(torch.sum(torch.sin(vx) ** 2)) * torch.sqrt(torch.sum(torch.sin(vy) ** 2)))
    return pearson_corr_real_val
    
def pearson_corr_complex(x , y):
    vx = x - torch.mean(x)
    vy = torch.conj(y) - torch.mean(torch.conj(y))
    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum((vx.abs())**2)) * torch.sqrt(torch.sum((vy.abs())**2)))
    return pearson_corr.abs()

def pearson_corr_complex_2(x , y):
    vx = x - torch.mean(x)
    vy = torch.conj(y) - torch.mean(torch.conj(y))
    vyy = y - torch.mean(y)   
    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum((vx.abs())**2)) * torch.sqrt(torch.sum((vyy.abs())**2)))
    return pearson_corr.abs()


##############################################################################################################
############################################# PLOTTING FUNCTIONS #############################################
##############################################################################################################

#Plots 1 image without showing it. This is a helper function used by the proceeding functions
def _imshow(x, locs=None, locs2=None, log=True, clim=None):
    #Bring to CPU
    x = x.detach().cpu()
    
    #Rectify x
    x = x.abs()
    
    #Do log
    if log:
        x = torch.log(x)
        
    #Plot
    plt.imshow(x)
    
        #Print locations
    if locs is not None:
        circle_rad = 5  # This is the radius, in points
        locs = locs.view(-1,2)
        for i in range(locs.shape[0]):
            plt.plot(locs[i,0],locs[i,1], 'x', ms=circle_rad * 2, mec='r', mfc='none', mew=2)
            
    #Print secondary locations if they exist
    if locs2 is not None:
        circle_rad = 5  # This is the radius, in points
        locs2 = locs2.view(-1,2)
        for i in range(locs2.shape[0]):
            plt.plot(locs2[i,0],locs2[i,1], 'x', ms=circle_rad * 2, mec='k', mfc='none', mew=2)
    
    #Set clim if not none
    if clim is not None:
        plt.clim(clim)

    plt.gca().invert_yaxis()
    
    #plt.axis('off')
    return


#Plots n images with n titles
def imshowN(LIST, TITLES, locs=None, locs2=None, log=True, n_rows=1, figsize_base=4, clim=None, colorbar=False):
    #Calculate number of columns
    n_cols = int(np.ceil(len(LIST)/n_rows))
    
    #Set locs
    if locs is None:
        locs = [None]*len(LIST)
    if locs2 is None:
        locs2 = [None]*len(LIST)
    
    #Start plotting
    plt.figure(figsize=[figsize_base*n_cols,figsize_base*n_rows])    
    for i in range(len(LIST)):
        plt.subplot(n_rows,n_cols, i+1)
        _imshow(LIST[i], locs=locs[i], locs2=locs2[i], log=log, clim=clim)
        plt.title(TITLES[i])
        if colorbar:
            plt.colorbar()
    plt.show()
    return

#Plots n-1 images, ignoring the first entry. This is used for models which don't use the impulse input (see impulse_input_from_loc) above
def imshowNLinear(LIST, TITLES, n_rows=1, figsize_base=4):
     imshowN(LIST[1:], TITLES[1:], n_rows=n_rows, figsize_base=figsize_base)

#Plots n images with n titles
def imshowN_same_cb(LIST, TITLES, locs=None, locs2=None, log=True, n_rows=1, figsize_base=4, clim=None, colorbar=False):
    #Calculate number of columns
    n_cols = int(np.ceil(len(LIST)/n_rows))
    
    #Set locs
    if locs is None:
        locs = [None]*len(LIST)
    if locs2 is None:
        locs2 = [None]*len(LIST)
    
    #Start plotting
    plt.figure(figsize=[figsize_base*n_cols,figsize_base*n_rows])    
    for i in range(len(LIST)):
        plt.subplot(n_rows,n_cols, i+1)
        _imshow(LIST[i], locs=locs[i], locs2=locs2[i], log=log, clim=clim)
        plt.title(TITLES[i])
    if colorbar:
        plt.colorbar()
    plt.show()
    return        


def plot_special_3D_MRI(MRI, slices=[1, 5, 10, 15, 20, 30, -1]):
    plt.figure(figsize=[20,5])

    for i, slice in enumerate(slices):
        plt.subplot(1,len(slices),i+1)
        plt.imshow(MRI[slice,:,:])
        plt.axis('off')
    plt.show()


def plot_special_3D_E_field(E_field, log = True, Ant = 1, slices=[1, 5, 10, 15, 20, 30, -1]):
    plt.figure(figsize=[20,5])

    if log:
        for i, slice in enumerate(slices):
            plt.subplot(1, len(slices),i+1)
            plt.imshow(E_field[Ant, slice,:,:].abs().log10())
            plt.axis('off')
    else:
        for i, slice in enumerate(slices):
            plt.subplot(1, len(slices),i+1)
            plt.imshow(E_field[Ant, slice,:,:].abs())
            plt.axis('off')
    plt.show()

##############################################################################################################
############################################## SAVING FUNCTIONS ##############################################
##############################################################################################################

def imsaveN(LIST, TITLES, tag, path, figsize_base=4, log=True, clim=None):
    #Start plotting
    for i in range(len(LIST)):
        plt.figure(figsize=[figsize_base,figsize_base])    
        _imshow(LIST[i], log=log, clim=clim)
        
        #Remove white space
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        #Save figure and close
        plt.savefig(os.path.join(path, TITLES[i] + tag + '.png'), pad_inches=0.0)
        plt.close()
    return

def imsaveNLinear(LIST, TITLES, tag, path, figsize_base=4):
    imsaveN(LIST[1:], TITLES[1:], tag, path=path, figsize_base=figsize_base)
    return    











import fileIO
import skimage.io as skio
import numpy as np
import timeit
import pdb

class dataset(object):
    '''__init__ inputs:
    \n file_id: list of file_ids 
    \n batch_size=4, 
    \n patchsize=64, 
    \n npatches=10**4, 
    \n fgfrac=.5, 
    \n shuffle=True, 
    \n rotate=True, 
    \n flip=True
    '''
    def __init__(self, file_id, batch_size=4, patchsize=64, getpatch_algo='random',npatches=10**4, 
        fgfrac=.5, shuffle=True, rotate=True, flip=True, stride=(32,32), padding=True):
        
        #Paths
        base_path,rel_im_path,rel_lbl_path,rel_result_path = fileIO.set_paths()
        self.im_path = base_path + rel_im_path
        self.lbl_path = base_path + rel_lbl_path
        self.results_path = base_path + rel_result_path
        self.file_id = file_id

        #Parameters for input
        self.batch_size = batch_size
        self.patchsize = patchsize

        self.getpatch_algo = getpatch_algo

        #Parameters for random patch algorithm
        self.npatches_perfile = int(npatches/len(file_id))
        self.fgfrac = fgfrac

        #Parameters used by stride algorithm
        self.stride = stride
        self.padding = padding

        #Parameters for augmentation
        #See examples in /Users/fruity/Envs/Py3ML/lib/python3.6/site-packages/keras/preprocessing/image.py
        self.shuffle = shuffle
        self.rotate = rotate
        self.flip = flip

        #Data buffer list: All raw files are retained in memory. 
        #GPU only receives minibatches.
        self.im_buffer = list()
        self.lbl_buffer = list()
        return

    def load_im_lbl(self):
        #This loads label and image data into a list
        for f in self.file_id:
            self.im_buffer.append(skio.imread(self.im_path + f + '.tif')/255.)#<---- Choosing 1st out of 3 exact copies in channels
            self.lbl_buffer.append(skio.imread(self.lbl_path + f + '_labels.tif')/1.)#Division by 1. forces float type.
        return

    def get_patches(self):
        #Use data buffer to generate patches

        im_patches = np.zeros((0, self.patchsize, self.patchsize, 1))
        lbl_patches = np.zeros((0, self.patchsize, self.patchsize, 1))
        for f in range(len(self.im_buffer)):
            if self.getpatch_algo == 'random':
                im_this, lbl_this = getpatches_randwithfg(
                    im=self.im_buffer[f], lbl=self.lbl_buffer[f],
                    patchsize=self.patchsize,
                    npatches=self.npatches_perfile,
                    fgfrac=self.fgfrac)
            
            elif self.getpatch_algo == 'stride':
                im_this, lbl_this = getpatches_strides(
                    im=self.im_buffer[f], lbl=self.lbl_buffer[f],
                    patchsize=self.patchsize, 
                    stride = self.stride,
                    padding=self.padding)

            elif self.getpatch_algo == 'inpadded':
                im_this, lbl_this = get_inpadded_patches_strides(
                    im=self.im_buffer[f], lbl=self.lbl_buffer[f],
                    patchsize=self.patchsize, 
                    padding=self.padding)

            im_patches = np.append(im_patches, im_this, axis=0)
            lbl_patches = np.append(lbl_patches, lbl_this, axis=0)

        im_patches,lbl_patches = data_augment(im_patches, lbl_patches, rotate=self.rotate, flip=self.flip, shuffle=self.shuffle)
        return im_patches, lbl_patches


def data_augment(im_patches, lbl_patches, rotate=True, flip=True, shuffle=True):
    '''Rotations, flip, or shuffle im_patches and lbl_patches the same way.
    \n Output:
    \n im_patches and lbl_patches concatenated arrays
    '''

    if rotate:
        #Stack rotated patches and labels
        im_patches = np.concatenate(
            (im_patches,
                np.rot90(im_patches, 1, (1, 2)),
                np.rot90(im_patches, 2, (1, 2)),
                np.rot90(im_patches, 3, (1, 2))), axis=0)

        lbl_patches = np.concatenate(
            (lbl_patches,
                np.rot90(lbl_patches, 1, (1, 2)),
                np.rot90(lbl_patches, 2, (1, 2)),
                np.rot90(lbl_patches, 3, (1, 2))), axis=0)

    if flip and rotate:
        im_patches = np.concatenate(
            (im_patches, np.flip(im_patches, axis=1)), axis=0)

        lbl_patches = np.concatenate(
            (lbl_patches, np.flip(lbl_patches, axis=1)), axis=0)

    elif flip and not rotate:
        im_patches = np.concatenate(
            (im_patches,
                np.flip(im_patches, axis=1),
                np.flip(im_patches, axis=2)), axis=0)

        lbl_patches = np.concatenate(
            (lbl_patches,
                np.flip(lbl_patches, axis=1),
                np.flip(lbl_patches, axis=2)), axis=0)

    if shuffle:
        shuffle_ind = np.random.permutation(np.arange(im_patches.shape[0]))
        im_patches = im_patches[shuffle_ind]
        lbl_patches = lbl_patches[shuffle_ind]

    return im_patches, lbl_patches


def getpatches_randwithfg(im, lbl, patchsize=64, npatches=10, fgfrac=.5):
    """Create patches where at least 'fgfrac' fraction of patches have atleast one foreground pixel in them
    \n Assumes 'lbl' contains foreground = 1 and background = 0"""

    shape = np.shape(im)
    pad = int(round(patchsize/2))
    nfg = int(round(npatches*fgfrac))
    if nfg > 0:
        #Find position of all foreground pixels:
        #np.where() operation takes ~0.04 s on my cpu to processing a (2500 x 2500) array.
        #These are used to create patches where at least one pixel will be foreground
        xmid_fg, ymid_fg = np.where(lbl == 2)
        
        if np.size(xmid_fg)==0:
            xmid_fg = np.random.randint(pad,shape[0]-pad,size=[nfg,],dtype=int)
            ymid_fg = np.random.randint(pad,shape[1]-pad,size=[nfg,],dtype=int)


        #Introduce jitter allow foreground pixel to be anywhere (not only center) within the patch
        xmid_fg = xmid_fg + np.random.randint(-(pad-1), (pad-1), np.size(xmid_fg), dtype=int)
        ymid_fg = ymid_fg + np.random.randint(-(pad-1), (pad-1), np.size(ymid_fg), dtype=int)
        ind = np.random.randint(0,np.size(xmid_fg),size=npatches,dtype=int)
        
        xmid_fg = xmid_fg[ind[0:nfg]]
        ymid_fg = ymid_fg[ind[0:nfg]]
    else:
        xmid_fg = np.array([],dtype = int)
        ymid_fg = np.array([],dtype = int)

    #Create random positions around which patches will be calculated
    xmid_rand = np.random.randint(pad,shape[0]-pad,size=[npatches-nfg,],dtype=int)
    ymid_rand = np.random.randint(pad,shape[1]-pad,size=[npatches-nfg,],dtype=int)
    
    #Concatenate patch locations
    xmid = np.concatenate((xmid_fg,xmid_rand), axis=0)
    ymid = np.concatenate((ymid_fg,ymid_rand), axis=0)
    
    #Calculating indices on image and on patch for dim 0 (x - coordinate)
    fxs = xmid - pad
    fxe = xmid + pad
    
    pxs = np.zeros(npatches,dtype=int)
    pxe = np.ones(npatches,dtype=int)*patchsize
    
    pxs[fxs<0] = abs(fxs[fxs<0])
    pxe[fxe>shape[0]] = patchsize - (fxe[fxe>shape[0]] - shape[0])

    fxs[fxs<0]=0
    fxe[fxe>shape[0]]=shape[0]

    #Repeating calculation for dim 1 (y - coordinate)
    fys = ymid - pad
    fye = ymid + pad
    
    pys = np.zeros(npatches,dtype=int)
    pye = np.ones(npatches,dtype=int)*patchsize

    pys[fys<0] = abs(fys[fys<0])
    pye[fye>shape[1]] = patchsize - (fye[fye>shape[1]] - shape[1])
    
    fys[fys<0]=0
    fye[fye>shape[1]]=shape[1]

    #Assemble patches
    im_patches = np.zeros([npatches,patchsize,patchsize,1])
    lbl_patches = np.zeros([npatches,patchsize,patchsize,1])
    for i in range(0, npatches):
        im_patches[i,pxs[i]:pxe[i],pys[i]:pye[i],0] = im[fxs[i]:fxe[i],fys[i]:fye[i]]
        lbl_patches[i,pxs[i]:pxe[i],pys[i]:pye[i],0] = lbl[fxs[i]:fxe[i],fys[i]:fye[i]]
    
    return im_patches, lbl_patches


def getpatches_strides(im, lbl, patchsize=64, stride = (32, 32),padding=True):
    """Returns im_patches and lbl_patches by striding across the image.
    \n im, lbl have the same size.
    \n stride: determines overlap between patches.
    \n padding=False will discard patches that don't fully lie within the image (assumes patchsize > stride)
    """
    shape = im.shape
    if padding:
        nstrides = np.round(np.array(shape)/np.array(stride))
    else:
        nstrides = np.round((np.array(shape)-patchsize)/np.array(stride))
    nstrides = nstrides.astype(int)
    npatches = nstrides[0]*nstrides[1]

    #Determine xy-coordinates of the patches
    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxe = fxs + patchsize

    pxs = np.zeros(np.shape(fxs),dtype=int)
    pxe = np.ones(np.shape(fxe),dtype=int)*patchsize
    
    pxs[fxs<0] = abs(fxs[fxs<0])
    pxe[fxe>shape[0]] = patchsize - (fxe[fxe>shape[0]] - shape[0])

    fxs[fxs<0]=0
    fxe[fxe>shape[0]]=shape[0]

    #Same for y-coordinate
    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fye = fys + patchsize
    
    pys = np.zeros(np.shape(fys),dtype=int)
    pye = np.ones(np.shape(fye),dtype=int)*patchsize

    pys[fys<0] = abs(fys[fys<0])
    pye[fye>shape[1]] = patchsize - (fye[fye>shape[1]] - shape[1])
    
    fys[fys<0]=0
    fye[fye>shape[1]]=shape[1]
    
    ij = 0
    im_patches = np.zeros([npatches,patchsize,patchsize,1])
    lbl_patches = np.zeros([npatches,patchsize,patchsize,1])
    for i in range(len(fxs)):
        for j in range(len(fys)):
            im_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = im[fxs[i]:fxe[i],fys[j]:fye[j]]
            lbl_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = lbl[fxs[i]:fxe[i],fys[j]:fye[j]]
            ij += 1
    return im_patches, lbl_patches


def combine_patches(xpatch,patchsize=64,stride=(64,64),imsize=(2500,2500),padding=True):
    '''xpatch is a 4 D array. 
    '''
    shape=imsize
    if padding:
        nstrides = np.round(np.array(shape)/np.array(stride))
    else:
        nstrides = np.round((np.array(shape)-patchsize)/np.array(stride))
    nstrides = nstrides.astype(int)

    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxe = fxs + patchsize

    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fye = fys + patchsize

    X = np.zeros([fxe[-1],fye[-1]])

    ij = 0
    for i in range(len(fxs)):
        for j in range(len(fys)):
            X[fxs[i]:fxe[i],fys[j]:fye[j]] = xpatch[ij,:,:,0]
            ij += 1

    return X
    



def get_inpadded_patches_strides(im, lbl, inner_pad=12, patchsize=64, padding=True):
    """Returns im_patches and lbl_patches by striding across the image.
    \n im, lbl have the same size.
    \n stride: determines overlap between patches.
    \n padding=False will discard patches that don't fully lie within the image (assumes patchsize > stride)
    """
    stride = (patchsize-inner_pad*2,patchsize-inner_pad*2)
    im_padded = np.pad(im,((inner_pad,inner_pad),),mode='constant',constant_values=(0.,))
    lbl_padded = np.pad(lbl,((inner_pad,inner_pad),),mode='constant',constant_values=(0.,))

    shape = im_padded.shape
    if padding:
        nstrides = np.floor(np.array(shape)/np.array(stride)) + 1
    else:
        nstrides = np.round((np.array(shape)-patchsize)/np.array(stride))
    nstrides = nstrides.astype(int)
    npatches = nstrides[0]*nstrides[1]

    #Determine xy-coordinates of the patches
    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxe = fxs + patchsize

    pxs = np.zeros(np.shape(fxs),dtype=int)
    pxe = np.ones(np.shape(fxe),dtype=int)*patchsize
    
    pxs[fxs<0] = abs(fxs[fxs<0])
    pxe[fxe>shape[0]] = patchsize - (fxe[fxe>shape[0]] - shape[0])

    fxs[fxs<0]=0
    fxe[fxe>shape[0]]=shape[0]

    #Same for y-coordinate
    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fye = fys + patchsize
    
    pys = np.zeros(np.shape(fys),dtype=int)
    pye = np.ones(np.shape(fye),dtype=int)*patchsize

    pys[fys<0] = abs(fys[fys<0])
    pye[fye>shape[1]] = patchsize - (fye[fye>shape[1]] - shape[1])
    
    fys[fys<0]=0
    fye[fye>shape[1]]=shape[1]
    
    ij = 0
    im_patches = np.zeros([npatches,patchsize,patchsize,1])
    lbl_patches = np.zeros([npatches,patchsize,patchsize,1])
    for i in range(len(fxs)):
        for j in range(len(fys)):
            im_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = im_padded[fxs[i]:fxe[i],fys[j]:fye[j]]
            lbl_patches[ij,pxs[i]:pxe[i],pys[j]:pye[j],0] = lbl_padded[fxs[i]:fxe[i],fys[j]:fye[j]]
            ij += 1
    return im_patches, lbl_patches


def combine_inpadded_patches(xpatch,inner_pad=12,patchsize=64,imsize=(2500,2500),padding=True):
    '''xpatch is a 4 D array. 
    '''
    
    inner_patchsize = patchsize-inner_pad*2
    stride = (inner_patchsize,inner_patchsize)

    shape=(imsize[0]+2*inner_pad,imsize[1]+2*inner_pad)
    if padding:
        nstrides = np.floor(np.array(shape)/np.array(stride)) + 1
    else:
        nstrides = np.round((np.array(shape)-patchsize)/np.array(stride))
    nstrides = nstrides.astype(int)

    fxs = np.arange(0, nstrides[0], 1) * stride[0]
    fxe = fxs + inner_patchsize

    fys = np.arange(0, nstrides[1], 1) * stride[1]
    fye = fys + inner_patchsize

    X = np.zeros([fxe[-1],fye[-1]])

    ij = 0
    for i in range(len(fxs)):
        for j in range(len(fys)):
            X[fxs[i]:fxe[i],fys[j]:fye[j]] = xpatch[ij,inner_pad:-inner_pad,inner_pad:-inner_pad,0]
            ij += 1

    return X

def debug_scripts():
    import im3dscroll as I
    D = dataset(['53'], batch_size=4, patchsize=500, getpatch_algo='stride',npatches=2, 
    fgfrac=.5, shuffle=False, rotate=True, flip=True, stride=(500,500), padding=True)
    D.load_im_lbl()
    im,lbl = D.get_patches()
    I.im3dscroll(im.reshape(im.shape[0],im.shape[1],im.shape[2]))
    I.im3dscroll(lbl.reshape(lbl.shape[0],lbl.shape[1],lbl.shape[2]))
    return

def debug_combine_patches():
    import im3dscroll as I
    import matplotlib.pyplot as plt

    D = dataset(['53'], batch_size=4, patchsize=64, getpatch_algo='stride',
        shuffle=False, rotate=True, flip=True, stride=(64,64), padding=True)
    D.load_im_lbl()

    im,lbl = D.get_patches()
    X = combine_patches(im,patchsize=D.patchsize,stride=D.stride,imsize=D.im_buffer[0].shape,padding=True)
    plt.ion()
    plt.imshow(X)
    #I.im3dscroll(im.reshape(im.shape[0],im.shape[1],im.shape[2]))
    #I.im3dscroll(lbl.reshape(lbl.shape[0],lbl.shape[1],lbl.shape[2]))
    return
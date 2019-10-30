# Yubei Chen, Sparse Manifold Coding Lib Ver 0.1
"""
This file contains multiple utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from numpy import random


def imshow(im,ax=None,nonBlock=False, title=None, vmin=None, vmax=None):
    axflag = True
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
        axflag = False
    axp = ax.imshow(im,cmap='gray',interpolation='none', vmin=vmin, vmax=vmax)
    if title!=None:
        ax.set_title(str(title))
    if (~nonBlock) and (not axflag):
        cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        cb = plt.colorbar(axp, cax = cbaxes)

def displayVecArry(basis,X=1,Y=1,ax='none',title=-1,nonBlock=False, equal_contrast = False,boundary='none'):
    axflag = True
    if ax == 'none':
        fig = plt.figure()
        ax = fig.gca()
        axflag = False
    basisTemp = basis.copy()
    if equal_contrast:
        #basis_mean = basisTemp.mean(1)
        for i in range(basisTemp.shape[-1]):
            basisTemp[:,i] = basisTemp[:,i] - basisTemp[:,i].mean()
            basisTemp[:,i] = basisTemp[:,i]/(basisTemp[:,i].max()-basisTemp[:,i].min())

    if len(basisTemp.shape) == 1:
        basisTemp = np.reshape(basisTemp,[basisTemp.shape[0],1])
    #if nonBlock:
    #    plt.ion()
    #else:
    #    plt.ioff()
    SHAPE = basisTemp.shape
    PATCH_SIZE = int(np.sqrt(SHAPE[0]))
    img = np.empty([(PATCH_SIZE+1)*X-1,(PATCH_SIZE+1)*Y-1])
    img.fill(np.min(basisTemp))
    if boundary != 'none':
        img.fill(np.min(boundary))
    for i in range(X):
        for j in range(Y):
            img[(PATCH_SIZE+1)*i:(PATCH_SIZE+1)*(i+1)-1,\
                (PATCH_SIZE+1)*j:(PATCH_SIZE+1)*(j+1)-1] = \
                np.reshape(basisTemp[:,i*Y+j],[PATCH_SIZE,PATCH_SIZE])
    ax.imshow(img,cmap='gray',interpolation='none')
    if title!=-1:
        ax.set_title(str(title))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")
    #time.sleep(0.05)
    #if ~nonBlock:
    #    plt.show()

def plotFrameOff(ax):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")

def createImage_FromVecArry(basis,X=1,Y=1):
    basisTemp = basis.copy()
    if len(basisTemp.shape) == 1:
        basisTemp = np.reshape(basisTemp,[basisTemp.shape[0],1])
    SHAPE = basisTemp.shape
    PATCH_SIZE = int(np.sqrt(SHAPE[0]))
    img = np.empty([(PATCH_SIZE+1)*X-1,(PATCH_SIZE+1)*Y-1])
    img.fill(np.min(basisTemp))
    for i in range(X):
        for j in range(Y):
            img[(PATCH_SIZE+1)*i:(PATCH_SIZE+1)*(i+1)-1,\
                (PATCH_SIZE+1)*j:(PATCH_SIZE+1)*(j+1)-1] = \
                np.reshape(basisTemp[:,i*Y+j],[PATCH_SIZE,PATCH_SIZE])
    return img

def displayVec3D(basisFunction, ax = 'none', title=-1, nonBlock=False):
    """
    Show a single basis function in wireframe, the basis function need to be
    reshaped into the right size
    """
    axflag = True
    if ax == 'none':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        axflag = False
    #if nonBlock:
    #    plt.ion()
    #else:
    #    plt.ioff()
    SHAPE = (basisFunction.T).shape
    #figure = plt.figure(fig)
    y = range(SHAPE[0])
    x = range(SHAPE[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, basisFunction.T)
    if (~nonBlock) and (not axflag):
        fig.show()
    #plt.show()

def displayVecSeq(VecSeq):
    return


def displayFourierRecenter(fourierIm, ax = 'none', nonBlock=False):
    axflag = True
    if ax == 'none':
        fig = plt.figure()
        ax = fig.gca()
        axflag = False
    dim0Shift = np.int(np.floor((fourierIm.shape[0]-1)/2.))
    dim1Shift = np.int(np.floor((fourierIm.shape[1]-1)/2.))
    tempIm = np.roll(fourierIm,dim0Shift,axis=0)
    tempIm = np.roll(tempIm,dim1Shift,axis=1)
    ax.imshow(tempIm,interpolation='none', cmap = 'gray')
    if (~nonBlock) and (not axflag):
        fig.show()
    return tempIm


def saveBasisParas(filename, _basis, _basis_size, _lambd, _iterations):
    np.savez(filename, basis=_basis, basis_size=_basis_size, lambd=_lambd, \
             iterations = _iterations)


def saveLattice(filename, _basis, _basis_size, _manifoldCoordinate, _settingParas):
    np.savez(filename, basis=_basis, basis_size=_basis_size, \
             manifoldCoordinate=_manifoldCoordinate, settingParas=_settingParas)


def loadLattice(filename):
    loaded = np.load(filename)
    return loaded['basis'], loaded['basis_size'], loaded['manifoldCoordinate'], loaded['settingParas']


def loadBasisParas(filename):
    loaded = np.load(filename)
    return loaded['basis'], loaded['basis_size'], loaded['lambd'], loaded['iterations']


def normalizeL2(vecArry):
    for i in range(vecArry.shape[1]):
        vecArry[:,i] = vecArry[:,i]/la.norm(vecArry[:,i],2)
    return vecArry

def normalizeL1(vecArry):
    for i in range(vecArry.shape[1]):
        vecArry[:,i] = vecArry[:,i]/la.norm(vecArry[:,i],1)
    return vecArry

def boundLinf(vecArry,bound):
    for i in range(vecArry.shape[1]):
        for j in range(vecArry.shape[0]):
            vecArry[j,i] = np.sign(vecArry[j,i])*np.min([np.abs(vecArry[j,i]),bound])
    return vecArry

def seq_filtering(Seq, filt):
    # This function temporally filter a time series
    # Each column of Seq is vector at a particular time step
    #TODO: Please finish this simple function
    return -1


def sampleRandom(data, num):
    """
    Currently data can not be 1d array
    """
    dataSize = data.shape
    dataNum = dataSize[-1]
    sampleSize = np.array(dataSize)
    sampleSize[-1] = num
    sample = np.zeros(sampleSize)
    batch = np.floor(random.rand(num)*dataNum)
    batch = batch.astype(np.int)
    for i in range(num):
        sample[:,i] = data[:,batch[i]]
    return sample

def sampleRandomWithParas(data,paras,num):
    """
    Currently data can not be 1d array
    """
    dataSize = data.shape
    paraSize = paras.shape
    dataNum = dataSize[-1]
    sampleSize = np.array(dataSize)
    sampleParaSize = np.array(paraSize)
    sampleSize[-1] = num
    sampleParaSize[-1] = num
    sample = np.zeros(sampleSize)
    sampleParas = np.zeros(sampleParaSize)
    batch = np.floor(random.rand(num)*dataNum)
    batch.astype(np.int)
    for i in range(num):
        sample[:,i] = data[:,batch[i]]
        sampleParas[:,i] = paras[:,batch[i]]
    return sample, sampleParas


def errorMsg(msg):
    """
    This function will output an error message msg
    """
    try:
        raise Exception(msg)
    except Exception as inst:
        print(inst)

def generalized_norm_square(V1,M):
    """
    This function returns a vector of square of the generalized norm of each columns in V1 with respect to M.
    """
    return np.diag(np.dot(V1.T,np.dot(M,V1)))

def generalized_norm(V1,M):
    """
    This function returns a vector of the generalized norm of each columns in V1 with respect to M.
    """
    return np.sqrt(np.diag(np.dot(V1.T,np.dot(M,V1))))

def patch_translation(patch,xshift,yshift):
    """
    Apply some pixel level translation on a given patch. It is not an in-place function.
    """
    xdim = patch.shape[0]
    ydim = patch.shape[1]
    patch_new = patch.copy()
    patch_new[...] = 0
    locx = np.clip(-xshift,0,np.Infinity).astype('int')
    locy = np.clip(-yshift,0,np.Infinity).astype('int')
    locx_new = np.clip(xshift,0,np.Infinity).astype('int')
    locy_new = np.clip(yshift,0,np.Infinity).astype('int')
    patch_new[locx_new:xdim+xshift,locy_new:ydim+yshift] = patch[locx:xdim-xshift,locy:ydim-yshift]
    return patch_new



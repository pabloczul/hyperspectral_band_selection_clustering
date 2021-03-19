import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat

def read_HSI():
    X = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    y = loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    #print(f"X shape: {X.shape}\ny shape: {y.shape}")
    return X, y

def h_mask(s, b, bins):
    '''
    This function takes in a matrix b representing a hyperspectral band and applies a sxs histogram mask to it.
    @s sixe of sxs mask
    @b band to apply.
    '''
    (width_b, heigth_b) = b.shape
    
    
    if s >= width_b or s>= heigth_b:
        return 0
    
    pad = (s-1)//2
    
    output = np.zeros((width_b-s+1, heigth_b-s+1,bins))
    h = np.zeros((bins))
    
    for y in np.arange(pad, heigth_b - pad):
        for x in np.arange(pad, width_b - pad):
            # extract the RegionOfInterest of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = b[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            (h,_) = np.histogram(roi, bins=bins)
            #histogram normalization
            #print(h)
            #print(np.linalg.norm(h))
            h = h/np.linalg.norm(h)
            #print(h)
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad,:] = h
                   
    return output

def hyper_filter(H_image, bins = 10, filter_size = 7):
    '''
    applies h_mask to a hyperspectra image

    @H_image: hyperspectral image to be processed
    '''
    (Wc, Hc, Bands) = H_image.shape

    Hyper_histogram_filter = np.zeros((Wc-filter_size+1, Hc-filter_size+1,Bands,bins))

    for i in range(Bands):
        Hyper_histogram_filter[:,:,i,:] = h_mask(filter_size,corrected[:,:,i], bins)

    return Hyper_histogram_filter

def bhattacharyya(dist_a,dist_b):
    '''
    Calculates battacharyya distance between distributions a and b (dist_a & dist b)
    '''
    #print("dista_pre---> ",dist_a)
    #print("distb_pre---> ",dist_b)
    dist_a = dist_a**(1/2)
    #print("dista--->",dist_a)
    dist_b = dist_b**(1/2)
    #print("distb---> ",dist_b)
    m=np.multiply(dist_a,dist_b)
    #print("m---> ",m)
    s=1-np.sum(m)
    #print("s---> ",s)
    bhatt=np.abs(s)**(1/2)
    #print("Bhatt: ",bhatt)
    
    return bhatt

def bhatt_integrate(band1, band2):
    '''
    Calculate distance between two histogram bands using func "bhattacharyya" and 
    ads them to get the value of correlation between two bands
    
    '''
    
    (height, width, _) = band1.shape
    
    integrate = 0
    
    for i in range(height):
        
        for j in range(width):
            
            integrate += bhattacharyya(band1[i,j],band2[i,j])
            #print("valor integral: ", integrate)
    return integrate

def bhatt_simil_matrix(hhm):
    '''
    
    @hhm hyperspectral histrogram matrix. matrix of the form (xpixels, ypixels, #bands, histogramSize)
    '''
    #Taking dimension out of hhm. 
    (height, width, bands,_) = hhm.shape
    #declaration of the correlation matrix, which must be #bandsX#bands. 
    #TODO: Change matrix for dictionary
    Corr_m = np.zeros((bands, bands))
    
    #iterates over resulting correlation matrix to save result of each band comparison
    for i in range(bands):
        for j in range(i+1, bands):
            if i == j :
                continue
            Corr_m[i,j] = bhatt_integrate(hhm[:,:,i,:],hhm[:,:,j,:])
    
    return Corr_m

if __name__ == '__main__':

    #read hyperspectral data 
    (corrected,ground_truth) = read_HSI()

    Bhatt_correlation_M = bhatt_simil_matrix(hyper_filter(corrected))

    np.savetxt('Bhatt_correlation_M.txt', Bhatt_correlation_M)
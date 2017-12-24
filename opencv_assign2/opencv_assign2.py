import cv2
import numpy as np
from matplotlib import pyplot as plt
import statistics
import time

# Histogram Calculation

def make_me_some_hists(input_path, output_path):
    #read image
    img = cv2.imread(input_path, 0)
    #make histograms
    histogram = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    cumulative_histogram = np.cumsum(histogram)
    #normalize
    histogram /= np.max(histogram)
    cumulative_histogram /= np.max(cumulative_histogram)
    #plot and export as image
    fig,ax = plt.subplots()
    ax.patch.set_facecolor('black')
    ax.bar(range(0,256), histogram, color = 'white', alpha=0.25)
    ax.bar(range(0,256), cumulative_histogram, color = 'white', alpha=0.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(10.24, 5.12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=100)
    return

def histogramCalculation():
    make_me_some_hists('input/cameraman.png', 'output/cameraman_hists.png')
    make_me_some_hists('input/bat.png', 'output/bat_hists.png')
    make_me_some_hists('input/fog.png', 'output/fog_hists.png')
    make_me_some_hists('input/fognoise.png', 'output/fognoise_hists.png')
    return

# Mean Verses Gaussian

def meanVersesGaussian():
    cameraman = cv2.imread('input/cameraman.png',0)
    cameraman_mean = cv2.blur(cameraman,(5,5))
    cameraman_gaussian = cv2.GaussianBlur(cameraman,(5,5),0)
    cv2.imwrite('output/cameraman_mean.png',cameraman_mean)
    cv2.imwrite('output/cameraman_gaussian.png',cameraman_gaussian)
    make_me_some_hists('output/cameraman_mean.png', 'output/cameraman_mean_hists.png')
    make_me_some_hists('output/cameraman_gaussian.png', 'output/cameraman_gaussian_hists.png')
    return

# Selective Median Filter

def fivebyfive_median_filter_pre(img):
    out = np.zeros(img.shape[:2])

    for i in range(2,len(img)-2):
        for j in range(2,len(img[i])-2):
            temp = []
            temp.append(img[i][j])
            temp.append(img[i-1][j])
            temp.append(img[i-2][j])
            temp.append(img[i][j-1])
            temp.append(img[i][j-2])
            temp.append(img[i+1][j])
            temp.append(img[i+2][j])
            temp.append(img[i][j+1])
            temp.append(img[i][j+2])
            temp.append(img[i-1][j-1])
            temp.append(img[i-2][j-2])
            temp.append(img[i-1][j-2])
            temp.append(img[i-2][j-1])
            temp.append(img[i+1][j+1])
            temp.append(img[i+2][j+2])
            temp.append(img[i+1][j+2])
            temp.append(img[i+2][j+1])
            temp.append(img[i+1][j-1])
            temp.append(img[i-2][j+2])
            temp.append(img[i-1][j+1])
            temp.append(img[i+2][j-2])
            temp.append(img[i+1][j-2])
            temp.append(img[i-2][j+1])
            temp.append(img[i-1][j+2])
            temp.append(img[i+2][j-1])
            out[i][j] = statistics.median(temp)        
    return out

def get_img_histogram_outliers(img,diff):
    img_histogram = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    not_zeros = []
    for index, element in enumerate(img_histogram):
        if int(element) != 0:
            not_zeros.append(index)
    median = statistics.median(not_zeros)
    outliers = []
    for element in not_zeros:
        if (element > (median+diff)) or (element < (median-diff)):
            outliers.append(element)

    return outliers

def fivebyfive_median_filter_post(img,outliers):
    out = np.zeros(img.shape[:2])

    for i in range(2,len(img)-2):
        for j in range(2,len(img[i])-2):
            if int(img[i][j]) in outliers:
                temp = []
                temp.append(img[i][j])
                temp.append(img[i-1][j])
                temp.append(img[i-2][j])
                temp.append(img[i][j-1])
                temp.append(img[i][j-2])
                temp.append(img[i+1][j])
                temp.append(img[i+2][j])
                temp.append(img[i][j+1])
                temp.append(img[i][j+2])
                temp.append(img[i-1][j-1])
                temp.append(img[i-2][j-2])
                temp.append(img[i-1][j-2])
                temp.append(img[i-2][j-1])
                temp.append(img[i+1][j+1])
                temp.append(img[i+2][j+2])
                temp.append(img[i+1][j+2])
                temp.append(img[i+2][j+1])
                temp.append(img[i+1][j-1])
                temp.append(img[i-2][j+2])
                temp.append(img[i-1][j+1])
                temp.append(img[i+2][j-2])
                temp.append(img[i+1][j-2])
                temp.append(img[i-2][j+1])
                temp.append(img[i-1][j+2])
                temp.append(img[i+2][j-1])
                out[i][j] = statistics.median(temp)
            else:
                out[i][j] = img[i][j]        
    return out

def selectiveMedianFilter():
    fognoise = cv2.imread('input/fognoise.png',0)
    start_pre = time.time()
    fognoise_median_pre = fivebyfive_median_filter_pre(fognoise)
    print("time taken pre enhancement: %s seconds " % (time.time() - start_pre))
    cv2.imwrite('output/fognoise_median_pre.png',fognoise_median_pre)

    start_post = time.time()
    fognoise_median_post = fivebyfive_median_filter_post(fognoise, get_img_histogram_outliers(fognoise,50))
    print("time taken post enhancement: %s seconds " % (time.time() - start_post))
    cv2.imwrite('output/fognoise_median_post.png',fognoise_median_post)
    return

# Contrast Stretching

def contrastStretching():
    frostfog = cv2.imread('input/frostfog.png',0)
    
    a = 0
    b = 255
    c = int(np.amin(frostfog))
    d = int(np.amax(frostfog))

    scaling_factor = (b-a)/(d-c)

    frostfog_cs = cv2.add(frostfog, -c)
    frostfog_cs = cv2.multiply(frostfog_cs, scaling_factor)
    frostfog_cs = cv2.add(frostfog_cs, a)

    cv2.imwrite('output/frostfog_cs.png',frostfog_cs)
    make_me_some_hists('output/frostfog_cs.png', 'output/frostfog_cs_hists.png')

    return

# Histogram Equalization

def histogramEqualization():
    frostfog = cv2.imread('input/frostfog.png',0)
    frostfog_histogram = cv2.calcHist([frostfog], [0], None, [256], [0,256]).flatten()
    frostfog_cumulative_histogram = np.cumsum(frostfog_histogram)
    c = int(np.amin(frostfog))
    min_freq = frostfog_cumulative_histogram[c]
    height, width = frostfog.shape[:2]
    no_of_pixels = height*width
    frostfog_he = np.zeros(frostfog.shape[:2])
    for i in range(len(frostfog)):
        for j in range(len(frostfog[i])):
            frostfog_he[i][j] = (frostfog_cumulative_histogram[int(frostfog[i][j])]-min_freq)*(255/(no_of_pixels-min_freq))
    cv2.imwrite('output/frostfog_he.png',frostfog_he)
    make_me_some_hists('output/frostfog_he.png', 'output/frostfog_he_hists.png')
    return

# Main

def main():
    histogramCalculation()
    meanVersesGaussian()
    selectiveMedianFilter()
    contrastStretching()
    histogramEqualization()
    return

# Program

main()
import cv2
import numpy as np

# Anaglyph
def createAnaglyph(left,right):
    anaglyph = np.zeros(left.shape, np.uint8)
    b,g,r = cv2.split(anaglyph)
    b = left[:,:,0]
    g = left[:,:,1]
    r = right[:,:,2]
    anaglyph = cv2.merge((b, g, r))
    return anaglyph

# Disparity
def createMinSADsImage(img1, img2, size, drmin, drmax, dcmin, dcmax):
    minSADs = np.zeros(img1.shape[:2])
    height, width = img1.shape[:2]
    mn_min = -1*(int(size/2))
    mn_max = int(size/2)
    for row in range(height):
        for col in range(width):
            SADs = []
            for dr in range(drmin,drmax+1):
                for dc in range(dcmin,dcmax+1):
                    SAD = 0
                    for m in range (mn_min, mn_max+1):
                        for n in range (mn_min, mn_max+1):
                            try:
                            	if (row+m)>=0 and (col+n)>=0 and (row+m+dr)>=0 and (col+n+dc)>=0:
                                	SAD += abs(int(img1[row+m][col+n])-int(img2[row+m+dr][col+n+dc]))
                            except IndexError:
                                pass
                    SADs.append(SAD)        
            minSADs[row][col] = min(SADs)
    minSADs = minSADs / np.max(minSADs) * 255
    return minSADs

# Main Func
def main():
    #reading the stereo pair - rgb
    left = cv2.imread('input/imL.png')
    right = cv2.imread('input/imR.png')
    #create anaglyph image
    anaglyph = createAnaglyph(left,right)
    cv2.imwrite('output/anaglyph.png', anaglyph)
    #reading the stereo pair - greyscale
    left = cv2.imread('input/imL.png', 0)
    right = cv2.imread('input/imR.png', 0)
    #create min SADs Image
    minSADs = createMinSADsImage(left,right,7,-1,1,-55,-45)
    cv2.imwrite('output/minSADs.png', minSADs)
    return

# Program
main()
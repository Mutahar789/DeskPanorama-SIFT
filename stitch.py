import cv2
import numpy as np

def stitch(img1, img2):
    # convert to gray scale and normalize
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    norm1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    norm2 = cv2.normalize(gray2, None, 0, 255, cv2.NORM_MINMAX)
    h1, w1 = norm1.shape
    h2, w2 = norm2.shape

    # find descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(norm1, None)
    kp2, des2 = sift.detectAndCompute(norm2, None)
    
    # compute distances
    distances = np.zeros((des1.shape[0], des2.shape[0]))
    for r,d1 in enumerate(des1):
        distances[r] = np.sqrt(np.sum(((des2-d1)**2), axis=1))
    
    # pair descriptors together
    matches = []
    for i in range(len(des1)):
        min = distances[i].min()
        j = distances[i].argmin()
        matches.append((min, i, j))

    # select 100 points
    matches = sorted(matches)
    src_pts = np.float32([kp1[i].pt for min,i,j in matches[:25]])
    dst_pts = np.float32([kp2[j].pt for min,i,j in matches[:25]])

    # find homography
    H, _ = cv2.findHomography(src_pts, dst_pts)
    translation = np.array([
        [1, 0, w1],
        [0, 1, 100],
        [0, 0,  1]
    ])
    H = np.matmul(translation, H)

    # warp
    result=cv2.warpPerspective(norm1, H, (w1+w2+50,max(h1,h2)+200))

    # combine
    positioned_2 = np.uint8(np.zeros((max(h1,h2)+200, w1+w2+50)))
    positioned_2[100:h1+100,w1:w1+w2] = norm2
    alpha = 0.5
    beta = 1-alpha
    final = cv2.addWeighted(result, 0.5, positioned_2, beta, 0.0)
    return final

def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img1 = cv2.resize(cv2.imread("Figure1.jpeg"), (650, 350))
img2 = cv2.resize(cv2.imread("Figure2.jpeg"), (650, 350))
stitched_img = stitch(img1, img2)
print("Displaying image 1...")
show(img1)
print("Displaying image 2...")
show(img2)
print("Displaying stitched image...")
show(stitched_img)

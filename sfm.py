import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_extract(images):
    '''
    Input: N images
    Output: Set of feature descriptors for each image
    '''
    print('Extracting features')
    read_images = []
    for i in range(len(images)):
        img = cv2.imread(images[i])
        read_images.append(img)
    assert len(read_images) >= 2, 'At least 2 images are needed'
    img1, img2 = read_images[0], read_images[1]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print('Extraction finished')
    return gray1, gray2, kp1, kp2, des1, des2

def feature_match(img1, img2, kp1, kp2, des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i]=[1,0]
            good_matches.append(m)
    draw_params = dict(matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = cv2.DrawMatchesFlags_DEFAULT)
    # breakpoint()
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # plt.imshow(img3,)
    # plt.show()
    return good_matches


def estimate_essential_matrix(kp1, kp2, matches, camera_matrix):
    pts1 = np.int32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.int32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    matchesMask = mask.ravel().tolist()
    return E, matchesMask   

def decompose_essential_matrix(E):
    # Perform SVD on the Essential matrix
    U, Sigma, Vt = np.linalg.svd(E)

    # Assume the camera moved forward
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Two possible rotation matrices
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)

    # Two possible translation vectors
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Return all four possible camera poses
    return [R1, t1], [R1, t2], [R2, t1], [R2, t2]

if __name__ == '__main__':
    image_list = ['1.jpg', '4.jpg']
    img1, img2, kp1, kp2, des1, des2 = feature_extract(images=image_list)
    matches = feature_match(img1, img2, kp1, kp2, des1, des2)
    camera_matrix = np.array([[1121.39684524,   0,        970.50628997],
                              [   0,         1121.39684524,    0.01171924],
                              [   0,            0,            1       ]])
    essential_matrix, mask = estimate_essential_matrix(kp1, kp2, matches, camera_matrix)
    poses = decompose_essential_matrix(essential_matrix)
    breakpoint()
    print('Script finished')


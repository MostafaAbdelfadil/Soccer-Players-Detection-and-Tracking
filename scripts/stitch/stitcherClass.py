import cv2
from cv2 import INTER_NEAREST
import numpy as np
import matplotlib.pyplot as plt

class Stitcher:
    def __init__(self):
        self.H = []
        self.bbox = None
        self.mask = [None, None]
        self.blurSize = 201 #odd number for gaussian blur to work properly

    def histogramGradient(self, img, axis):
        hist = np.sum(img, axis=axis)
        hist = hist / np.max(hist)
        #returns the absolute of the gradient since we're only interested in the magnitude of the change
        return np.absolute(np.gradient(hist))

    def getMask(self, img):
        mask = np.zeros((img.shape[0], self.blurSize))
        mask[:, :int(self.blurSize/3)] = 1
        mask = cv2.GaussianBlur(mask, ksize=(self.blurSize, self.blurSize), sigmaX=0, sigmaY=0)
        mask = np.dstack((mask, mask, mask))
        return mask

    def trim(self, img):
        #calculating the bbox
        if self.bbox is None:
            margin = 50
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist_x = self.histogramGradient(gray, axis=0)
            midpoint = img.shape[1] // 2
            xlow, xhigh = np.argmax(hist_x[:midpoint]), midpoint + np.argmax(hist_x[midpoint:])
            hist_y = self.histogramGradient(gray, axis=1)
            midpoint = img.shape[0] // 2
            ylow, yhigh = np.argmax(hist_y[:midpoint]), midpoint + np.argmax(hist_y[midpoint:])
            self.bbox = [xlow, xhigh, ylow-margin, yhigh+margin]

        #unpacking the bbox
        xlow, xhigh, ylow, yhigh = self.bbox
        return img[ylow:yhigh, xlow:xhigh]                


    def stitch(self, images, showMatches=False, crop=True):
        if len(images) == 2:
            return self.trim(self.stitchPair(images, 0))

        #stitch mid and right
        left, mid, right = images
        pair = [mid, right]
        intermediate_img = self.stitchPair(pair, 0)

        #flip result and flip left image
        intermediate_img = cv2.flip(intermediate_img, 1)
        left = cv2.flip(left, 1)

        #stitch flipped result with flipped left
        pair = [intermediate_img, left]
        result = self.stitchPair(pair, 1)

        #flip again to get original image stitched then trim it
        result = cv2.flip(result, 1)
        if crop: result = self.trim(result)
        return result

    
    #takes a pair of images in images list and stitches them
    def stitchPair(self, images, ind, ratio=0.75, reprojThreshold=4.0, smoothing=True, showMatches=False):
        #B -> dst, A -> src
        imageB, imageA = images
        #haven't computed a homography matrix yet
        if len(self.H) <= ind:
            kpsA, featuresA = self.detectAndDescribe(imageA)
            kpsB, featuresB = self.detectAndDescribe(imageB)
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThreshold)

            if M is None:
                return None
            
            matches, _, status = M
            self.H.append(M[1])

        #warp perspective and stitch
        result = cv2.warpPerspective(imageA, self.H[ind], (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        #masking
        if smoothing:
            if self.mask[ind] is None:
                self.mask[ind] = self.getMask(imageB)

            imageB[:, -self.blurSize:] = (imageB[:, -self.blurSize:] * self.mask[ind]).astype(np.uint8)
            result[:imageB.shape[0], imageB.shape[1]-self.blurSize:imageB.shape[1]] = (
                result[:imageB.shape[0], imageB.shape[1]-self.blurSize:imageB.shape[1]] * (1 - self.mask[ind])
            ).astype(np.uint8)
            result[:imageB.shape[0], :imageB.shape[1]-self.blurSize] = imageB[:, :imageB.shape[1]-self.blurSize]
            result[:imageB.shape[0], imageB.shape[1]-self.blurSize:imageB.shape[1]] += imageB[:, -self.blurSize:]
        else: 
            result[:imageB.shape[0], :imageB.shape[1]] = imageB

        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            cv2.imshow('matching', vis)
            cv2.waitKey(0)
        return result
    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (255, 0, 0), 1)
        # return the visualization
        return vis
    
    def detectAndDescribe(self, image):
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThreshold):
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for _, i in matches])
            ptsB = np.float32([kpsB[i] for i, _ in matches])

            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThreshold)
            return matches, H, status

        return None

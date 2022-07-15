import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine

BINS=4

def getHistogram(img, mask=None):
    '''returns normalized RGB histogram of img'''
    hist = np.zeros((BINS*3,), dtype=np.float128)
    if mask is None:
        h, w, c = img.shape
        mask = np.ones((h,w))
    
    for c in range(3):
        hist[c*BINS:(c+1)*BINS], _ = np.histogram(img[:,:,c][mask>0], bins=BINS, range=(0,256))
    
    hist = hist / np.sum(hist)
    #plt.plot(hist)
    return hist



class Detection:
    def __init__(self, bbox, hist, masry_dist=None, masry_gk_dist=None, pyramids_dist=None, pyramids_gk_dist=None, referee_dist=None, centroid=None, area=None):
        self.hist = hist
        self.bbox = bbox
        self.centroid = centroid
        self.area = area
        self.masry_dist = masry_dist
        self.masry_gk_dist = masry_gk_dist
        self.pyramids_dist = pyramids_dist
        self.pyramids_gk_dist = pyramids_gk_dist
        self.referee_dist = referee_dist

class Detector:
    def __init__(self):
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=5)
        self.masry_histogram = np.load('masry.npy')
        self.pyramids_histogram = np.load('pyramids.npy')
        self.masry_gk_histogram = np.load('masry_gk.npy')
        self.pyramids_gk_histogram = np.load('pyramids_gk.npy')
        self.referee_histogram = np.load('referee.npy')
        self.masry_color = (255,255,255)
        self.pyramids_color = (0,0,0)
        self.masry_gk_color = (255,255,255)
        self.pyramids_gk_color = (0,0,0)
        self.threshold = 0.11
        self.learning_rate = 0
        self.area_thresh = 450
        self.ROI = cv2.imread('ROI.png', flags=cv2.IMREAD_GRAYSCALE)



    def detectPlayers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        mask = self.bg_subtractor.apply(gray, learningRate=0.1)
        raw_mask = mask.copy()
        #plt.imshow(mask, cmap='gray')
        #mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)[1]
        #mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((2,2)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5)), iterations=2)

        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9)))
        #plt.imshow(mask, cmap='gray')
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        
        if numLabels < 3: return mask
        detections = self.filterCC(numLabels, labels, stats, centroids, img, mask)
        for detection in detections: self.drawDetection(img, detection.bbox, (255,0,0))
        return mask

    def getTeams(self, img, bboxes, model='resnet'):
        '''returns a dict of both teams and referee.'''
        detections = self.filterDetections(img, bboxes)
        res = {
            'referee': None,
            'masry': [],
            'pyramids': []
        }
        #get the referee if detected
        detections = sorted(detections, key=lambda x: x.referee_dist)
        if detections[0].referee_dist < self.threshold:
            res['referee'] = detections[0]
            self.updateHistogram(detections[0].hist, team='referee')
            del detections[0]

        #get masry gk if detected
        detections = sorted(detections, key=lambda x: x.masry_gk_dist)
        if detections[0].masry_gk_dist < self.threshold:
            res['masry'] = [detections[0]]
            self.updateHistogram(detections[0].hist, team='masry_gk')
            del detections[0]

        #get pyramids gk if detected
        detections = sorted(detections, key=lambda x: x.pyramids_gk_dist)
        if detections[0].pyramids_gk_dist < self.threshold:
            res['pyramids'] = [detections[0]]
            self.updateHistogram(detections[0].hist, team='pyramids_gk')
            del detections[0]

        #get masry players
        detections = sorted(detections, key=lambda x: x.masry_dist)
        for i, detection in enumerate(detections):
            if detection.masry_dist > self.threshold: 
                detections = detections[i:]
                break
            res['masry'].append(detection)
            #only update with the four highest confidences
            if i < 4:
                self.updateHistogram(detection.hist, team='masry')

        #get pyramids players
        detections = sorted(detections, key=lambda x: x.pyramids_dist)
        for i, detection in enumerate(detections):
            if detection.pyramids_dist > self.threshold: break
            res['pyramids'].append(detections[i])
            #only update with the four highest confidences
            if i < 4:
                self.updateHistogram(detections[i].hist, team='pyramids')
        
        if model == 'resnet':
            res = self.convertBboxes(res)
        return res
        
    def convertBboxes(self, detections):
        res = {
            'referee': None,
            'masry': [],
            'pyramids': []
        }
        x,y,w,h = detections['referee'].bbox
        res['referee'] = x,y, x+w, y+h
        for i, _ in enumerate(detections['masry']):
            x,y,w,h = detections['masry'][i].bbox
            res['masry'].append((x,y,x+w,y+h))

        for i, _ in enumerate(detections['pyramids']):
            x,y,w,h = detections['pyramids'][i].bbox
            res['pyramids'].append((x,y,x+w,y+h))

        return res

    
    def filterDetections(self, img, bboxes):
        '''rejects detections outside the ROI.'''
        detections = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            if self.ROI[y2,x1] > 0:
                x,y,w,h = x1, y1, x2-x1, y2-y1
                patch = img[y:y+h, x:x+w]
                hist = getHistogram(patch)
                masry_gk = cosine(hist, self.masry_gk_histogram)
                pyramids_gk = cosine(hist, self.pyramids_gk_histogram)
                referee = cosine(hist, self.referee_histogram)
                masry = cosine(hist, self.masry_histogram)
                pyramids = cosine(hist, self.pyramids_histogram)
                detections.append(Detection(
                    (x,y,w,h), hist, masry_dist=masry, masry_gk_dist=masry_gk, pyramids_dist=pyramids, pyramids_gk_dist=pyramids_gk, referee_dist=referee
                    ))
        return detections
    
    def filterCC(self, numLabels, labels, stats, centroids, img, mask):
        '''returns filtered detections.'''
        detections = []
        masry = []
        pyramids = []
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50 and h > 1.2*w and self.ROI[min(629,y+h),x] > 0:
                detection = Detection(
                    (x,y,w,h), hist=None, centroid=centroids[i], area=area
                    )
                detections.append(detection)
        return detections


    def drawDetection(self, img, bbox, color):
        x,y,w,h = bbox
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

    def updateHistogram(self, hist, team):
        if team == 'masry':
            self.masry_histogram = (1-self.learning_rate) * self.masry_histogram + self.learning_rate * hist
        elif team == 'pyramids':
            self.pyramids_histogram = (1-self.learning_rate) * self.pyramids_histogram + self.learning_rate * hist
        elif team == 'masry_gk':
            self.masry_gk_histogram = (1-self.learning_rate) * self.masry_gk_histogram + self.learning_rate * hist
        elif team == 'pyramids_gk':
            self.pyramids_gk_histogram = (1-self.learning_rate) * self.pyramids_gk_histogram + self.learning_rate * hist
        elif team=='referee':
            self.referee_histogram = (1-self.learning_rate) * self.referee_histogram + self.learning_rate * hist


    def _filterCC(self, numLabels, labels, stats, centroids):
        '''returns bbox and centroids of players and filters out irrelevant labels.'''
        good_bbox = []
        good_centroids = []
        good_area = []
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if (area > 25 and h > 1.1*w):
                good_bbox.append((x,y,w,h))
                good_centroids.append(centroids[i])
                good_area.append(area)
        return good_bbox, good_centroids, good_area


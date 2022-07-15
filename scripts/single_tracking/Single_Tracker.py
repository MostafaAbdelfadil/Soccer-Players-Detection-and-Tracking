import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from TopView import PerspectiveTransformation
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

#histogram parameters
mask = None
channels = [0,2]
histSize = [32]*2
ranges = [0,255]*2

#meanshift termination criteria
term_crit = (cv2.TermCriteria_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

class Tracker:
    def __init__(self, id):
        self.id = id
        self.roi_hist = None
        self.track_window = None
        self.positions = []
        self.df = None
        self.transform = PerspectiveTransformation()
        self.M = self.transform.M

    
    def calcHist(self, frame):
        self.track_window = cv2.selectROI('Match', frame, showCrosshair=False)
        x,y,w,h = self.track_window
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame[y:y+h, x:x+w]
        images = [roi]
        self.roi_hist = cv2.calcHist(images, channels, mask, histSize, ranges)
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    def apply(self, frame, register=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        back_proj = cv2.calcBackProject([frame], channels, self.roi_hist, ranges, 1)
        num_iters, self.track_window = cv2.meanShift(back_proj, self.track_window, term_crit)
        x,y,w,h = self.track_window
        if register:
            pts = np.float32(np.array([[[x, y+self.margin(y)]]]))  
            new = cv2.perspectiveTransform(pts, self.M)[0]
            x_new, y_new = new[0].astype(int)
            #interpolating to plotting coordinates
            x_new = ((x_new-95)/(1405-95)) * 130
            y_new = 90 - ((y_new-85)/(911-85)) * 90
            self.positions.append((x_new, y_new))
        return self.track_window, back_proj
    
    def margin(self, y):
        m_min, m_max = 16, 31
        y_min, y_max = 66, 485
        slope = (m_max - m_min) / (y_max - y_min)
        margin = slope * (y - y_min) + m_min
        return int(margin)
    
    def save(self):
        self.df = pd.DataFrame(np.array(self.positions), columns=['x', 'y'])
        self.df.to_json(self.id + '.json')
        self.saveHeatmap()
        
    def saveHeatmap(self):
        #creating heatmap
        #Create figure
        fig=plt.figure()
        fig.set_size_inches(7, 5)
        ax=fig.add_subplot(1,1,1)

        #Pitch Outline & Centre Line
        plt.plot([0,0],[0,90], color="black")
        plt.plot([0,130],[90,90], color="black")
        plt.plot([130,130],[90,0], color="black")
        plt.plot([130,0],[0,0], color="black")
        plt.plot([65,65],[0,90], color="black")

        #Left Penalty Area
        plt.plot([16.5,16.5],[65,25],color="black")
        plt.plot([0,16.5],[65,65],color="black")
        plt.plot([16.5,0],[25,25],color="black")

        #Right Penalty Area
        plt.plot([130,113.5],[65,65],color="black")
        plt.plot([113.5,113.5],[65,25],color="black")
        plt.plot([113.5,130],[25,25],color="black")

        #Left 6-yard Box
        plt.plot([0,5.5],[54,54],color="black")
        plt.plot([5.5,5.5],[54,36],color="black")
        plt.plot([5.5,0.5],[36,36],color="black")

        #Right 6-yard Box
        plt.plot([130,124.5],[54,54],color="black")
        plt.plot([124.5,124.5],[54,36],color="black")
        plt.plot([124.5,130],[36,36],color="black")

        #Prepare Circles
        centreCircle = plt.Circle((65,45),9.15,color="black",fill=False)
        centreSpot = plt.Circle((65,45),0.8,color="black")
        leftPenSpot = plt.Circle((11,45),0.8,color="black")
        rightPenSpot = plt.Circle((119,45),0.8,color="black")

        #Draw Circles
        ax.add_patch(centreCircle)
        ax.add_patch(centreSpot)
        ax.add_patch(leftPenSpot)
        ax.add_patch(rightPenSpot)

        #Prepare Arcs
        leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
        rightArc = Arc((119,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")

        #Draw Arcs
        ax.add_patch(leftArc)
        ax.add_patch(rightArc)

        #Tidy Axes
        plt.axis('off')

        sns.kdeplot(data=self.df, x="x", y="y", shade=True,n_levels=200, fill=True, cmap='ch:start=2, rot=0, dark=0.01, light=1')
        plt.ylim(0, 90)
        plt.xlim(0, 130)

        plt.savefig(f'{self.id}.png')

import numpy as np
import cv2 as cv

# This class is meant to moduleize the detection algorithm I implemented
class ConeDetector:
    '''
    Contains a method to detect the path marked by red cones in an image.
    '''

    def __call__(self, img_dir):
        '''
        the function takes an image directory as input, generate an 'answer.png' as output in the current working directory, 
        the original image should contain a straight path defined by red cones, while the answer.png file will have the boundaries 
        marked by two straight red lines

        Parameters
        ------------------
        img_dir : the directory of the image we want to process
        '''
        # process image
        image = cv.imread(img_dir)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV) # convert the image to hsv to better separate red from the rest of the image
        lower_red1 = np.array([0, 140, 150]) # lower range for red hue, saturation and value tuned to better isolate the cones
        upper_red1 = np.array([2, 255, 200])

        lower_red2 = np.array([170, 250, 150]) # upper range for red hue
        upper_red2 = np.array([180, 250, 200])

        # Threshold the image to get only red colors
        mask1 = cv.inRange(hsv_image, lower_red1, upper_red1) # red region from lower range
        mask2 = cv.inRange(hsv_image, lower_red2, upper_red2) # red region from upper range
        red_mask = cv.bitwise_or(mask1, mask2) # combine the two masks

        # edge detection
        edges = cv.Canny(red_mask, 950, 1500)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # get contour coordinates

        # linear regression for the coordnates of the cones (which come from contours)
        contour_points = []
        for contour in contours: # process the contour coordinates so that only one average coordinate for each contour detected
            contour = np.squeeze(contour, axis=1) # too many unnessesary dimensions for the previous contour
            # get the average x and y of all contour coordinates
            x = (sum(contour[:, 0])/len(contour))
            y = (sum(contour[:, 1])/len(contour))
            contour_points.append([x, y])

        # use k-means clustering to get the left and right halves of the path boundary
        _, labels, _ = cv.kmeans(np.float32(np.array(contour_points)), 2, None, (cv.TERM_CRITERIA_MAX_ITER, 100, 0), 10, cv.KMEANS_RANDOM_CENTERS)
        contour_points_1 = [] # we don't really know which one is right and which is left, only group 1 and group 2
        contour_points_2 = []
        for i, point in enumerate(labels): # assign contour points to the groups
            if point==1:
                contour_points_1.append(contour_points[i])
            else:
                contour_points_2.append(contour_points[i])
        contour_points_1 = np.array(contour_points_1) # convert to np array
        contour_points_2 = np.array(contour_points_2) # convert to np array
        # linear regressions
        line_left = cv.fitLine(contour_points_1, cv.DIST_L2, 0, reps=0.01, aeps=0.01)
        line_right = cv.fitLine(contour_points_2, cv.DIST_L2, 0, reps=0.01, aeps=0.01)

        # output image with the regression lines
        cv.line(image, (int(line_left[0]*10000+line_left[2]), int(line_left[1]*10000+line_left[3])), (int(line_left[2]-line_left[0]*10000), int(line_left[3]-line_left[1]*10000)), color=(255, 0, 0), thickness=2)
        cv.line(image, (int(line_right[0]*10000+line_right[2]), int(line_right[1]*10000+line_right[3])), (int(line_right[2]-line_right[0]*10000), int(line_right[3]-line_right[1]*10000)), color=(255, 0, 0), thickness=2)
        #cv.imshow('cones', image)
        #cv.imshow('edges' ,edges)
        #cv.waitKey(0)
        cv.imwrite('answer.png', image)

# a demo of the code using the sample image in the github directory of the coding challenge
detector = ConeDetector()
detector('.\\red.png')

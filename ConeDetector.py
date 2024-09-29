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
        upper_red1 = np.array([2, 255, 210])

        lower_red2 = np.array([170, 140, 150]) # upper range for red hue
        upper_red2 = np.array([180, 255, 210])

        # Threshold the image to get only red colors
        mask1 = cv.inRange(hsv_image, lower_red1, upper_red1) # red region from lower range
        mask2 = cv.inRange(hsv_image, lower_red2, upper_red2) # red region from upper range
        red_mask = cv.bitwise_or(mask1, mask2) # combine the two masks

        # edge detection
        edges = cv.Canny(red_mask, 950, 1500)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # get contour coordinates

        # linear regression for contours on the left half of the image and the right half
        contour_points_left = []
        contour_points_right = []
        for contour in contours: # separate contours into left half and right half of the image according to their x-coordinates
            contour = np.squeeze(contour, axis=1) # too many unnessesary dimensions for the previous contour
            # get the average x and y of all contour coordinates
            x = (sum(contour[:, 0])/len(contour))
            y = (sum(contour[:, 1])/len(contour))
            if x <= len(image[0])/2:
                contour_points_left.append([x, y])
            else:
                contour_points_right.append([x, y])
        contour_points_left = np.array(contour_points_left) # convert to np array
        contour_points_right = np.array(contour_points_right) # convert to np array
        # linear regressions
        line_left = cv.fitLine(contour_points_left, cv.DIST_L2, 0, reps=0.01, aeps=0.01)
        line_right = cv.fitLine(contour_points_right, cv.DIST_L2, 0, reps=0.01, aeps=0.01)

        # output image with regression lines
        '''
        for point in contour_points_left:
            aaa = (int(point[0]), int(point[1]))
            cv.circle(image, aaa, radius=5, color=(255, 0, 0), thickness=-1)
        for point in contour_points_right:
            aaa = (int(point[0]), int(point[1]))
            cv.circle(image, aaa, radius=5, color=(255, 0, 0), thickness=-1)
        '''
        cv.line(image, (int(line_left[0]*10000+line_left[2]), int(line_left[1]*10000+line_left[3])), (int(line_left[2]-line_left[0]*10000), int(line_left[3]-line_left[1]*10000)), color=(0, 0, 255), thickness=2)
        cv.line(image, (int(line_right[0]*10000+line_right[2]), int(line_right[1]*10000+line_right[3])), (int(line_right[2]-line_right[0]*10000), int(line_right[3]-line_right[1]*10000)), color=(0, 0, 255), thickness=2)
        cv.imshow('cones', image)
        cv.imshow('edges' ,edges)
        cv.waitKey(0)
        cv.imwrite('answer.png', image)

# a demo of the code using the sample image in the github directory of the coding challenge
detector = ConeDetector()
detector('.\\red.png')
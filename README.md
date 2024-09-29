# WACodeChallenge-Jason-Li
This is the application code challenge for the perception department of Wisconsin Autonomous
The ConeDetector class has a __call__() method that takes an image with a path marked by red cones as input, and generate an ansewr.png file with the bounaries of the path marked by red lines as output

## Libraries and the Enviorment
This python program uses python 3.9.19, opencv 4.10.0, and numpy 1.26.4

## Methodology
The program first convert the image to hsv format and threshold it with opencv to separate red colors from the image. The thresholds are tuned so that only the colors that matches the cones are separated out. Then, we use opencv again to find the edges and the coordinates of the contours of the cones. Each coordinate of the contours are averaged so that an approximate coordinate of the center of cones are found. We then classify the coordinates by their position either on the left half or on the right half of the image. Do a linear regression for each group to get two regression lines that marks the edge of the path. Afterwards, the image with the lines are output as "answer.png"

## Methods attempted and failed to work

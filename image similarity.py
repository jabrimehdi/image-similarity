#imports
#openCv Library
import cv2 as operation
#matplotlib 
#using pyplot for a Matlab-Like visualization
import numpy as np 
from matplotlib import pyplot as window

#change the PATHs
#uploading Images
picture1 = operation.imread('C:/../images/image1.jpg')
picture2 = operation.imread('C:/../images/image2.jpg')
picture3 = operation.imread('C:/../images/image3.jpg')

#calculate the histogram for each image
histogram1 = operation.calcHist([picture1], [0], None, [256], [0, 255])
histogram2 = operation.calcHist([picture2], [0], None, [256], [0, 255])
histogram3 = operation.calcHist([picture3], [0], None, [256], [0, 255])

#calculating similarity rate (based on histogram)
#similarity rate (between image 1 and 3) 
sim1 = operation.compareHist(histogram1, histogram3, operation.HISTCMP_INTERSECT)
#similarity rate (between image 2 and 3) 
sim2 = operation.compareHist(histogram2, histogram3, operation.HISTCMP_INTERSECT)

print('Similarity rate (Image 1 & 3) ', sim1)
print('Similarity rate (Image 2 & 3) ', sim2)

#displaying images
operation.imshow("Image 1", picture1)
operation.imshow("Image 2", picture2)
operation.imshow("Image 3", picture3)

#displaying histogram(s)
window.plot(histogram1)
window.plot(histogram2)
window.plot(histogram3)
window.title("Histograms")
window.show()

operation.waitKey(0)
operation.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread('../images/cars.jpg')
bbox, label, conf = cvimage.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)

plt.imshow(output_image)
plt.show()
print('Number of cars detected:' + str(label.count('car')))
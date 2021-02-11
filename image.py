# image.py - Primer detekcije objekata na jednoj slici koristeći YOLO v3 model
# (unapred istrenirani koeficijenti i konvertovani koeficijenti)

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
import os
import glob
import winsound
import math

model_size = (416, 416,3)   # Očekivani ulazni format za dati model i istrenirane koeficijente
num_classes = 80            # Broj klasa nad kojima je mreža trenirana  
class_name = './data/coco.names'    # Putanja do datoteke koja sadrži imena klasa
max_output_size = 40                # Najveći broj okvira koje želimo da dobijemo za sve klase ukupno
max_output_size_per_class= 20       # Najveći broj okvira koje želimo da dobijemo po klasi
iou_threshold = 0.5                 # Prag mere preklapanja dva okvira
confidence_threshold = 0.5          # Prag pouzdanosti prisustva objekta 

cfgfile = './cfg/yolov3.cfg'                  # Putanja do YOLO v3 konfiguracione datoteke
weightfile = './weights/yolov3_weights.tf'    # Putanja do datoteke koja sadrži istrenirane koeficijente u TensorFlow formatu
img_path_left_cam = "./data/images/image_L"        # Putanja do ulazne slike nad kojom se vrši detekcija
img_path_right_cam = "../image_R"

# sound duration in ms
cSoundDuration = 1000
# sound frequency
cSoundFrequency = 1000
# left camera id
cLeftCamId = 0
# right camera id 
cRightCamId = 1
# distance between cameras
cCameraDistance = 0.54
# image width
cImageWidth = 1762
# field of view 
# 0.872665 rad = 50 degrees
cFieldOfView = 0.872665 
#Objects of interest from coco.names classes
ObjectsOfInterest = [0, 1, 2, 3, 5, 6, 7]

def loadAndResize(imgsDir):
    print ('loading  images...')

    images = []
    resized_images = []
    filenames = []

    os.chdir(imgsDir)
    for imagePath in glob.glob("*.png"):
        img = cv2.imread(imagePath)
        img = np.array(img)

        images.append(img)

        #img = tf.expand_dims(img, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
        resized_images.append(resize_image(img, (model_size[0],model_size[1])))

        filenames.append(os.path.basename(imagePath))
    print ('loading complete!')

    return [images, resized_images, filenames]

def calculateDistance(disparity):
    return (cCameraDistance * cImageWidth) / (2 * math.tan(cFieldOfView / 2) * disparity)

def objectDistance(leftImg, rightImg, boxes, nums, classes):

    boxesLeft, numsLeft = boxes[0], nums[0]
    boxesRight, numsRight = boxes[1], nums[1]
    boxesLeft=np.array(boxesLeft)
    boxesRight = np.array(boxesRight)

    classesLeft, classesRight = classes[0], classes[1]

    print(classesLeft)
    print(classesRight)

    print('levi: ', nums[0])
    print('desni:', nums[1])
    
    j = 0
    for i in range(min(numsLeft,numsRight)):
        x1y1_leftImg = tuple((boxesLeft[i,0:2] * [leftImg.shape[1],leftImg.shape[0]]).astype(np.int32))
        x2y2_leftImg = tuple((boxesLeft[i,2:4] * [leftImg.shape[1],leftImg.shape[0]]).astype(np.int32))
        x1y1_rightImg = tuple((boxesRight[i,0:2] * [rightImg.shape[1],rightImg.shape[0]]).astype(np.int32))
        x2y2_rightImg = tuple((boxesRight[i,2:4] * [rightImg.shape[1],rightImg.shape[0]]).astype(np.int32))

        centrePointX_leftImg = (x1y1_leftImg[0] + x2y2_leftImg[0]) / 2
        centrePointX_rightImg = (x1y1_rightImg[0] + x2y2_rightImg[0]) / 2
        
        #print("cetrePointX_left: ", centrePointX_leftImg)
        #print("centrePointX_right: ", centrePointX_rightImg)
        
        #print("Disparity ", i, ":", abs(centrePointX_leftImg - centrePointX_rightImg)) 
        
        
        if(classesLeft[i] in ObjectsOfInterest):
            disparity = abs(centrePointX_leftImg - centrePointX_rightImg)
            distanceFromObject = calculateDistance(disparity)
        
            print("Distance from object: ", distanceFromObject)
        
def main():

    # Kreiranje modela
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    # Učitavanje istreniranih koeficijenata u model
    model.load_weights(weightfile)
    # Učitavanje imena klasa
    class_names = load_class_names(class_name)
	
	# Učitavanje ulaznih fotografija i predobrada u format koji očekuje model
    images_left = []
    resized_images_left = []
    filenames_left = []
    
    # Load left camera data 
    [images_left, resized_images_left, filenames_left] = loadAndResize(img_path_left_cam)
    
    images_right = []
    resized_images_right = []
    filenames_right = []
    
    # Load right camera data 
    [images_right, resized_images_right, filenames_right] = loadAndResize(img_path_right_cam)
    
    # Inferencija nad ulaznom slikom
    # izlazne predikcije pred - skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta 
    for i in range(0, len(filenames_left)):
        resized_image = []
        
        image = images_left[i]

        resized_image.append(resized_images_left[i])
        resized_image.append(resized_images_right[i])
        
        resized_image = tf.expand_dims(resized_image, 0)
        resized_image = np.squeeze(resized_image)
        
        pred = model.predict(resized_image)

        # Određivanje okvira oko detektovanih objekata (za određene pragove)
        boxes, scores, classes, nums = output_boxes( \
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)

        objectDistance(images_left[i], images_right[i], boxes, nums, classes)

        out_img = draw_outputs(image, boxes, scores, classes, nums, class_names, cLeftCamId)

        # Čuvanje rezultata u datoteku
        out_file_name = './out/Izlazna slika.png'
        cv2.imwrite(out_file_name, out_img)

        # Prikaz rezultata na ekran
        cv2.imshow(out_file_name, out_img)
        #cv2.waitKey(0)

        if(cv2.waitKey(20) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

# sound notification
#winsound.Beep(cSoundFrequency, cSoundDuration)

if __name__ == '__main__':
    main()
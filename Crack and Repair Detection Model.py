import torch
import pandas as pd
import math as m
import os
import cv2

path = './data/UAPD_RDD/img/test/' #change the path to where your sliced images are stored
#pixel_factor = 0.00428375  this is the parameter for converting the pixel lengths to true lengths for the drone survey conducted by us

#pixel factor, you can input these parameters and calculate the pixel factor for your sepcific image.
height = 2
focal_length = 47e-4
pixel_dim = 8e-7
pixel_factor = pixel_dim*(focal_length+  height)/focal_length
print("Pixel Factor: " ,pixel_factor)

detections = []
longitudinal_crack = 0
lateral_crack = 0
alligator_crack = 0
transverse_crack = 0
oblique_crack = 0
block_crack = 0
repair = 0
pothole = 0

def crack_width(img, crack_type):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    __ ,th2 = cv2.threshold(img,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h, w = th2.shape
    widths = []
    stops = []

    if crack_type == 2:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
    for i in range(h):
        pix_count = 0
        prev = 1
        for j in range(w):
            if th2[i][j] == 0 and prev == 0:
                pix_count += 1
            else:
                widths.append(pix_count)
                stops.append([i,j])
                pix_count = 0
            prev = th2[i][j]
    return max(widths), stops[widths.index(max(widths))]

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', _verbose =False) #store the weight for the yolo model in the same directory as this script under the name best.pt

for i in os.listdir(path):
    results = model(path+i)
    results.save('./') #this line is working, but creates individual directories for each inference, working on fixing it
    #save the entire images with bboxes
    results_crops = results.crop(save=False)
    for k in range(len(results_crops)):
        img = results_crops[k]["im"]
        distress_class = results_crops[k]["cls"]
        conf = results_crops[k]["conf"]
        bbox = results_crops[k]["box"]

        if distress_class == 0: #Longitudinal crack

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            width, __ = crack_width(img, distress_class)

            crack_length = pixel_factor*(ymax-ymin)
            detections.append([i , "Longitudinal crack", crack_length, "N.A", width*pixel_factor])
            longitudinal_crack +=1

        elif distress_class == 1: #Alligator crack

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            crack_area = pixel_factor*pixel_factor*((xmax-xmin)*(ymax-ymin))
            detections.append([i , "Alligator crack", "N.A", crack_area, "N.A"])
            alligator_crack +=1

        elif distress_class == 2: #Transverse crack

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            width, __ = crack_width(img, distress_class)

            crack_length = pixel_factor*(xmax-xmin)
            detections.append([i , "Tranvserse crack", crack_length, "N.A", width*pixel_factor])
            transverse_crack +=1

        elif distress_class == 3: #Oblique crack

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            crack_length = pixel_factor*(m.sqrt((xmax-xmin)**2+(ymax-ymin)**2))
            detections.append([i , "Oblique crack", crack_length, "N.A", "N.A"])
            oblique_crack +=1

        elif distress_class == 4: #Repair

            repair +=1
            detections.append([i , "Repair site", "N.A", "N.A", "N.A"])

        elif distress_class == 5: # Pothole

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            crack_area = (pixel_factor**2)*((xmax-xmin)*(ymax-ymin))
            detections.append([i , "Pothole crack", "N.A", crack_area, "N.A"])
            pothole +=1           
            

print("Number of Longitudinal cracks: ", longitudinal_crack)
print("Number of Lateral cracks: ", lateral_crack)
print("Number of Alligator cracks: ", alligator_crack)
print("Number of Transverse cracks: ", transverse_crack)
print("Number of Oblique cracks: ", oblique_crack)
print("Number of Block cracks: ", block_crack)
print("Number of Repair sites: ", repair)
print("Number of Potholes: ", pothole)
print(len(detections))

df = pd.DataFrame(detections, columns=["Image ID", "Class", "Length", "Area", "Width"])
print(df)
#df.to_excel("detections.xlsx")

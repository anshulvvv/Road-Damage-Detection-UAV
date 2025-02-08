# imports
from tqdm import tqdm
from matplotlib import patches
import numpy as np
from osgeo import gdal, osr
import rasterio as rio
import torch
import scipy
from scipy import misc
import matplotlib.pyplot as plt
from flask import *
#from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
import torch, pandas
import os
import logging
import math as m
logging.basicConfig(level=logging.INFO)
from werkzeug.utils import secure_filename
import jsons
import numpy as np
from osgeo import gdal, osr
from PIL import Image
from pymongo import MongoClient
from flask_cors import CORS
import math
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Initialisations
SLICE_SIZE = 512
MM_TO_PIXEL = 4.11  # Check from report needs to be set everytime
#PATH_TO_MODEL = '/content/drive/MyDrive/rdd-integration/best.pt' // updated below to reflect latest folder name
PATH_TO_MODEL = 'rdd_model/best.pt'
height = 2
focal_length = 47e-4
pixel_dim = 8e-7
pixel_factor = pixel_dim*height/focal_length
MM2_TO_M2 = 1000000
avgLat = 0.
avgLong = 0.
driver_name = "GTiff"
drivePath = 'rdd_model/'
# Road Health Index
minimum_distress_width = 1 # Specify the unit here

#Mongo db setup
client=MongoClient("mongodb+srv://<USERNAME>:<PASSWORD>@<CLUSTER>/")

db = client.get_database("distress_db")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Supporting classes
class Rdd:
    def init(self, roadID, roadHealthIndex, latitude, longitude, longitudinalCracks, transverseCracks, obliqueCracks, repairs, potholes):
      self.roadID = roadID
      self.roadHealthIndex = roadHealthIndex
      self.latitude = latitude
      self.longitude = longitude
      self.longitudinalCracks = longitudinalCracks
      self.transverseCracks = transverseCracks
      self.obliqueCracks = obliqueCracks
      self.repairs = repairs
      self.potholes = potholes
      return self

    def serialize(self):
      return {"_id" : self.roadID, "roadHealthIndex": self.roadHealthIndex, "meanLatitude": self.latitude, "meanLongitude": self.longitude, "longitudinalCracks": self.longitudinalCracks,
              "transverseCracks": self.transverseCracks, "obliqueCracks": self.obliqueCracks, "repairs": self.repairs, "potholes": self.potholes}

class RddList:
    def initList(self, list):
      self.list = list
      return self

    def serialize(self):
        return {"list": self.list}

class RoadDistress:
    def init(self, latitude, longitude, severity):
      self.latitude = latitude
      self.longitude = longitude
      self.severity = severity
      return self

    def serialize(self):
        return {"latitude": self.latitude, "longitude": self.longitude, "severity": self.severity}

# All functions
# convert the final bbox coords wrt to the ones on the DEM
def coord_converter(xmin_photo, ymin_photo, xmax_photo, ymax_photo, dataset_photo, dataset_dem, label):
    # Get geotransform of the orthophoto
    geotransform_photo = dataset_photo.GetGeoTransform()

    # Get the real world coordinates
    x_min_real = geotransform_photo[0] + xmin_photo * \
        geotransform_photo[1] + ymin_photo * geotransform_photo[2]
    y_min_real = geotransform_photo[3] + xmin_photo * \
        geotransform_photo[4] + ymin_photo * geotransform_photo[5]

    x_max_real = geotransform_photo[0] + xmax_photo * \
        geotransform_photo[1] + ymax_photo * geotransform_photo[2]
    y_max_real = geotransform_photo[3] + xmax_photo * \
        geotransform_photo[4] + ymax_photo * geotransform_photo[5]

    # use real word coords on the geotagged orthophoto
    geotransform_photo_dem = dataset_dem.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset_dem.GetProjection())

    coord_transform = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

    # Convert the real-world coordinates to projected coordinates
    x_proj_min, y_proj_min, _ = coord_transform.TransformPoint(
        x_min_real, y_min_real)
    x_proj_max, y_proj_max, _ = coord_transform.TransformPoint(
        x_max_real, y_max_real)

    td0 = geotransform_photo_dem[0]
    td1 = geotransform_photo_dem[1]
    td2 = geotransform_photo_dem[2]
    td3 = geotransform_photo_dem[3]
    td4 = geotransform_photo_dem[4]
    td5 = geotransform_photo_dem[5]

    x_min_dem = int(
        ((x_proj_min - td0)*td5 - (y_proj_min - td3)*td2)/(td1*td5 - td2*td4)
    )

    y_min_dem = int(
        ((x_proj_min - td0)*td4 - (y_proj_min - td3)*td1)/(td2*td4 - td1*td5)
    )

    x_max_dem = int(
        ((x_proj_max - td0)*td5 - (y_proj_max - td3)*td2)/(td1*td5 - td2*td4)
    )

    y_max_dem = int(
        ((x_proj_max - td0)*td4 - (y_proj_max - td3)*td1)/(td2*td4 - td1*td5)
    )

    # Note the last two items are the real world coordinates of the centroid of the bbox to identify and locate each pothole
    return x_min_dem, y_min_dem, x_max_dem, y_max_dem, (x_min_real + x_max_real)/2, (y_min_real + y_max_real)/2, label

def calculate_slice_bboxes(
    orthophoto_dataset,
    image_height: int,
    image_width: int,
    slice_height: int = SLICE_SIZE,
    slice_width: int = SLICE_SIZE,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2):

    def check_garbage_slice(dataset, xmin, ymin, xmax, ymax):

        image = np.array(dataset.ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin))
        image = image[0:3, :, :]
        corner1 = image[:, 0, 0]
        corner2 = image[:, ymax - ymin - 1, xmax - xmin - 1]
        corner3 = image[:, 0, xmax - xmin - 1]
        corner4 = image[:, ymax - ymin - 1, 0]
        reject = np.array([255, 255, 255])
        corners = [corner1, corner2, corner3, corner4]
        for corner in corners:
            if (corner == reject).all():
                return True
        return False

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                if not check_garbage_slice(orthophoto_dataset, xmin, ymin, xmax, ymax):
                    slice_bboxes.append([xmin, ymin, xmax, ymax])

            else:
                if not check_garbage_slice(orthophoto_dataset, x_min, y_min, x_max, y_max):
                    slice_bboxes.append([x_min, y_min, x_max, y_max])

            x_min = x_max - x_overlap
        y_min = y_max - y_overlap

    return slice_bboxes

def get_check_bbox_coords(bboxes, oboxes, result, image_dim, slice):

    xmin_slice, ymin_slice, xmax_slice, ymax_slice = slice
    boxes = result.xyxyn[0]
    for box in boxes:
        box = np.array(box.detach().cpu().clone().numpy())
        xmin, ymin, xmax, ymax, confidence, label = box
        xmin *= image_dim
        xmax *= image_dim
        ymin *= image_dim
        ymax *= image_dim

        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin += xmin_slice
        ymin += ymin_slice
        xmax += xmin_slice
        ymax += ymin_slice

        bbox = (xmin, ymin, xmax, ymax, label)

        # Potholes
        if label == 4 and confidence > 0.5:
            bboxes.append(bbox)
        # other distresses
        elif confidence > 0.5:
            oboxes.append(bbox)

def show_image_2(image):

    r, g, b = image[0], image[1], image[2]
    rgb_image = np.stack([r, g, b], axis=-1)
    plt.imshow(rgb_image)
    plt.savefig("/content/drive/MyDrive/extractedRddImages/imageWithAxes.png")

def show_image_without_axes(image,sliceCount):
    r, g, b = image[0], image[1], image[2]
    rgb_image = np.stack([r, g, b], axis=-1)

    # Create a figure without axes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')  # Turn off the axes

    # Display the image
    ax.imshow(rgb_image)

    # Save the image without axes
    filePath = "/content/drive/MyDrive/extractedRddImages/test" + str(sliceCount) + ".png"
    plt.savefig(filePath, bbox_inches='tight', pad_inches=0)

def show_image_2(image):

    r, g, b = image[0], image[1], image[2]
    #rgb_image = np.stack([r, g, b], axis=-1)
    rgb_image = np.stack([r, g, b])
    plt.imshow(rgb_image)
    plt.savefig("/content/drive/MyDrive/extractedRddImages/test1.png")

def detector(slices, dataset_ortho):
    bboxes = []
    oboxes = []
    print("Running the detector model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH_TO_MODEL, source='github')
    #print('Printing sliced images')
    #sliceCount = 0;
    for slice in tqdm(slices):
        xmin, ymin, xmax, ymax = slice
        sliced_image = np.array(dataset_ortho.ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin))
        sliced_image = sliced_image[0:3, :, :]
        #show_image_without_axes(sliced_image, sliceCount)
        #sliceCount = sliceCount+1
        result = model(sliced_image)
        get_check_bbox_coords(bboxes, oboxes, result, sliced_image.shape[1], slice)   # adds the bbox coordinates to the list wrt orthophoto
    return bboxes, oboxes

def convert_to_dem_coords(bboxes, allLat, allLong, dataset_ortho, dataset_dem):
    final_bboxes_dem = []
    for box in bboxes:
        final_box_dem = coord_converter(box[0], box[1], box[2], box[3], dataset_ortho, dataset_dem, box[4])
        final_bboxes_dem.append(final_box_dem)
        allLat.append(final_box_dem[4])
        allLong.append(final_box_dem[5])
    return final_bboxes_dem

def calculate_volume_and_maxdepth(final_bboxes_dem, dem_array, allLat, allLong):
    """
    Append the volume and maximum depth of the pothole to the list
    """
    volume_max_depth = []
    area = MM_TO_PIXEL*MM_TO_PIXEL

    for box in final_bboxes_dem:

        x_min_dem, y_min_dem, x_max_dem, y_max_dem, x_real, y_real, label = box
        sliced_dem = dem_array[y_min_dem:y_max_dem,
                                    x_min_dem:x_max_dem]
        max_depth = np.max(sliced_dem)
        min_depth = np.min(sliced_dem)

        max_depth_floor = (max_depth - min_depth)

        # We shall use our min depth as the floor depth
        volume = 0
        for i in range(sliced_dem.shape[0]):
            for j in range(sliced_dem.shape[1]):
                depth_wrt_floor = sliced_dem[i][j] - min_depth
                volume += area*depth_wrt_floor
        allLat.append(x_real)
        allLong.append(y_real)
        # contains the volume, the max depth, and the real world coords of the potholes
        volume_max_depth.append((volume, max_depth_floor, (x_real, y_real)))

    return volume_max_depth

def compute_distress_severity(obox_vals, alligatorCracks, longitudinalCracks, transverseCracks, obliqueCracks, repairs):

    for val in obox_vals:
      xmin = val[0]
      ymin = val[1]
      xmax = val[2]
      ymax = val[3]
      latitude = val[4]
      longitude = val[5]
      label = val[6]
      crack_length = 0.
      crack_area = 0.

      if label == 0: #Alligator crack
        crack_area = pixel_factor*pixel_factor*((xmax-xmin)*(ymax-ymin))
        vals_with_severity = [latitude, longitude, str(crack_area)]
        alligatorCracks.append(vals_with_severity)

      elif label == 1: #Longitudinal crack
        crack_length = pixel_factor*(ymax-ymin)
        vals_with_severity = [latitude, longitude, str(crack_length*minimum_distress_width)] # Road Health Index
        longitudinalCracks.append(vals_with_severity)

      elif label == 2: #Oblique crack
        crack_length = pixel_factor*(m.sqrt((xmax-xmin)**2+(ymax-ymin)**2))
        vals_with_severity = [latitude, longitude, str(crack_length*minimum_distress_width)] # Road Health Index
        obliqueCracks.append(vals_with_severity)

      #3 left out as a faulty typo'd class, 4/potholes left out as computed in separate function

      elif label == 5: #Repair
        repair_area = pixel_factor*pixel_factor*((xmax-xmin)*(ymax-ymin)) # Road Health Index
        vals_with_severity = [latitude, longitude, str(repair_area)]
        repairs.append(vals_with_severity)

      elif label == 6: #Transverse crack
        crack_length = pixel_factor*(xmax-xmin)
        vals_with_severity = [latitude, longitude, str(crack_length*minimum_distress_width)] # Road Health Index
        transverseCracks.append(vals_with_severity)


def calculate_severity(volume_max_depth, obox_vals_with_severity, potholes, bbox_vals):

    # using final bboxes ortho and not final bboxes dem for consistency across the mm to pixel ratio
    for i, box in enumerate(bbox_vals):

        xmin, ymin, xmax, ymax, label = box
        xlength = xmax - xmin
        ylength = ymax - ymin

        xlength *= MM_TO_PIXEL
        ylength *= MM_TO_PIXEL

        width = max(xlength, ylength)

        volume, depth, real_coords = volume_max_depth[i]

        # depth value is in metres so convert to mm
        depth *= 1000

        # Severity is either small or medium
        if width >= 500:
            if depth >= 25 and depth <= 50:
                vals_with_severity = [real_coords[0], real_coords[1], 'medium']

            elif depth > 50:
                vals_with_severity = [real_coords[0], real_coords[1], 'large']
            else:
                # Classify as medium in undefined edge case
                vals_with_severity = [real_coords[0], real_coords[1], 'medium']

        else:
            # Classify as small in all undefined edge cases in this category
            if depth <= 25:
                vals_with_severity = [real_coords[0], real_coords[1], 'small']
            else:
                vals_with_severity = [real_coords[0], real_coords[1], 'medium']

        potholes.append(vals_with_severity)

    return potholes

# Road Health Index
def calculate_road_health_index(roadLength, roadWidth, alligatorCracks,longitudinalCracks, transverseCracks, obliqueCracks, repairs, potholes):

  road_area = float(roadLength) * float(roadLength)
  crack_extent = (sum(float(item[2]) for item in alligatorCracks) + sum(float(item[2]) for item in longitudinalCracks) + sum(float(item[2]) for item in transverseCracks) + sum(float(item[2]) for item in obliqueCracks))/road_area
  pothole_count = len(potholes)
  patch_extent = sum(float(item[2]) for item in repairs)

  PCI_Cracking = 100 * (math.exp(-0.0534 * crack_extent) - 0.006641 * crack_extent)
  PCI_Pothole = (-130.6 * pothole_count**3 + 2112 * pothole_count**2 - 9550 * pothole_count + 14390) / (pothole_count**3 + 6.619 * pothole_count**2 - 66.73 * pothole_count + 144.1)
  PCI_Patch = 7231 / (patch_extent**2 - 0.737 * patch_extent + 73.09)

  return round(0.2222 * PCI_Pothole + 0.3333 * PCI_Cracking + 0.4444 * PCI_Patch, 2)

#Services
@app.route('/api/health', methods=['GET'])
def books():
        return "Service is healthy", 201

@app.route('/api/getRoadHealthIndex/<id>', methods=['GET'])
def getRoadHealthIndex(id):
  records = db.distress_records
  road_data = records.find_one({"_id" : id})
  if road_data is not None:
    return road_data, 201
  return jsonify({"message" : f"Road with ID {id} not found"}), 404

@app.route('/api/routes', methods=['GET'])
def routes():
  records = db.distress_records
  road_ids = records.find(projection=["_id"])
  if road_ids is not None:
    return list(road_ids)
  return jsonify({"message" : f"Road DB is empty"}), 404

@app.route('/api/delete/<id>', methods=['DELETE'])
def delete(id):
  records = db.distress_records
  delete_result = records.delete_one({"_id": id})

  if delete_result.deleted_count == 1:
      return jsonify({'message' : 'Successfully deleted'}), 200

  return jsonify({"message" : f"Failed to delete road {id}"}), 400

@app.route('/api/computeRoadHealthIndex', methods=['POST'])
def computeRoadHealthIndex():
    # validate the files
    os.makedirs(os.path.join(app.instance_path, 'htmlfi'), exist_ok=True)

    if 'roadID' not in request.form:
        resp = jsonify({'message': 'No road ID provided'})
        resp.status_code = 400
        return resp

    if 'roadLength' not in request.form:
        resp = jsonify({'message': 'No road length provided'})
        resp.status_code = 400
        return resp

    if 'roadWidth' not in request.form:
        resp = jsonify({'message': 'No road width provided'})
        resp.status_code = 400
        return resp

    if 'groundResVal' not in request.form:
        resp = jsonify({'message': 'No ground resolution value provided'})
        resp.status_code = 400
        return resp

    records = db.distress_records
    road_data = records.find_one({"_id" : request.form["roadID"]})
    if road_data is not None:
      return jsonify({"message" : f"Road ID already exists"}), 400

    if 'orthoFile' not in request.files:
        resp = jsonify({'message': 'No ortho file part in the request'})
        resp.status_code = 400
        return resp
    if 'depthFile' not in request.files:
        resp = jsonify({'message': 'No depth file part in the request'})
        resp.status_code = 400
        return resp

    # read and save the files
    orthoFile = request.files['orthoFile']
    PATH_TO_ORTHOPHOTO = os.path.join(drivePath, secure_filename(orthoFile.filename))
    orthoFile.save(PATH_TO_ORTHOPHOTO)

    depthFile = request.files['depthFile']
    PATH_TO_DEM = os.path.join(drivePath, secure_filename(depthFile.filename))
    depthFile.save(PATH_TO_DEM)

    #Initialize MM_TO_PIXEL
    MM_TO_PIXEL = request.form["groundResVal"]

    # Initialize distresses
    alligatorCracks = []
    longitudinalCracks = []
    transverseCracks = []
    obliqueCracks = []
    repairs = []
    potholes = []

    # Compute dem array
    dataset_ortho = gdal.Open(PATH_TO_ORTHOPHOTO)
    dataset_dem = gdal.Open(PATH_TO_DEM)
    dem_array = np.array(dataset_dem.ReadAsArray())

    # Compute slices
    slices = calculate_slice_bboxes(
        orthophoto_dataset=dataset_ortho,
        image_height=dataset_ortho.RasterYSize,
        image_width=dataset_ortho.RasterXSize
        )
    print('Slices length : ', len(slices))

    # Compute bbox val
    bbox_vals, obox_vals = detector(slices=slices, dataset_ortho=dataset_ortho)
    print('bbox_val length : ', len(bbox_vals))
    allLat = []
    allLong = []
    final_oboxes_dem = convert_to_dem_coords(obox_vals, allLat, allLong, dataset_ortho, dataset_dem)
    obox_vals_with_severity = compute_distress_severity(final_oboxes_dem, alligatorCracks, longitudinalCracks, transverseCracks, obliqueCracks, repairs)

    # Compute final bbox dem
    final_bboxes_dem = convert_to_dem_coords(bbox_vals, allLat, allLong, dataset_ortho, dataset_dem)

    # Compute volume max depth - working
    volume_max_depth = calculate_volume_and_maxdepth(final_bboxes_dem, dem_array, allLat, allLong)
    # Compute final results
    calculate_severity(volume_max_depth, obox_vals_with_severity, potholes, bbox_vals)

    avgLat = np.mean(allLat)
    avgLong = np.mean(allLong)

    # Road Health Index
    road_health_index = calculate_road_health_index(request.form["roadLength"], request.form["roadWidth"], alligatorCracks,longitudinalCracks, transverseCracks, obliqueCracks, repairs, potholes)

    RddObj = Rdd()
    pothole = RddObj.init(request.form["roadID"], road_health_index, avgLat, avgLong, jsons.dump(longitudinalCracks), jsons.dump(transverseCracks), jsons.dump(obliqueCracks), jsons.dump(repairs), jsons.dump(potholes))

    #Mongo db - save the distress
    records = db.distress_records
    records.insert_one(pothole.serialize())
    print('Added to db')
    print(pothole.serialize())
    return jsonify({'message' : 'Successfully processed'}), 201

app.run()
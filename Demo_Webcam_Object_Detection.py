# This is a code for real time object detection using different deep learning trained models in Tensorflow.
# We used a webcam for real time testing.
# You can find the full documentation and installation instructions at: 
# https://github.com/tensorflow/models/tree/master/research/object_detection

import tensorflow as tf
import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import urllib
import sys
import tarfile
import zipfile



# This is needed since all the main TF codes are stored in the object_detection folder.
sys.path.append("/home/mahdi/Tensorflow-1.5/tensorflow/models/research/object_detection/")
sys.path.append("/home/mahdi/Tensorflow-1.5/tensorflow/models/research/")

from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/mahdi/Tensorflow-1.5/tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90




#Download Model
# You would probably need this only the first time you want to download a new deep learning model.
# Tou can ignore it afterward.

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())









detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


IMAGE_SIZE = (12, 8)


cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.Session(graph = detection_graph) as sess:

      
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      tensor_dict = [boxes, scores, classes, num_detections]


      while(True):
        ret, image_np = cap.read()

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


		# Run inference
		# Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})


        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          # instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)

        cv2.imshow('frame',cv2.resize(image_np, (800,600)))

        # You can exit by pressing q
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


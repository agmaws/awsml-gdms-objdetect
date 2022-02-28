import matplotlib.pyplot as plt
import matplotlib.patches as patches    
import matplotlib.image as mpimg
import random
import numpy as np
import json
import cv2
import boto3
from botocore.errorfactory import ClientError
s3client = boto3.client('s3')
s3 = boto3.resource("s3")
object_categories = ['Person', 'StackedBoxes', 'Forklift']

def visualize_manifest(manifest_record):
    dt = json.loads(augmented_manifest_lines[80])
    visualize_detection('image_0000247.5.png',[(0,1,1075/1920,381/1080,(1075+128)/1920,(274+381)/1080)],['a','b','c'],0.1)

def visualize_detection(img_file, dets, classes=[], thresh=0.6):
    """
    visualize detections in one image
    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    bucket = s3.Bucket('ml-materials')
    img1 = bucket.Object(f'object_detection_dataset/source_images/{img_file}').get().get('Body').read()
    imageRGB = cv2.imdecode(np.asarray(bytearray(img1)), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imageRGB , cv2.COLOR_BGR2RGB)
    #img=mpimg.imread(img_file)
    f, ax = plt.subplots(1, 1)
    ax.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    output = []
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        cls_id = int(klass)
        class_name = str(cls_id)
        if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
        output.append([class_name, score])
        if score < thresh:
            continue
        colors[0] = (1, 0, 0)   
        colors[1] = (0, 1, 0)
        colors[2] = (0, 0, 1)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        rect = patches.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=0.5)
        ax.add_patch(rect)


        ax.text(xmin, ymin - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                          fontsize=12, color='white')

    return f, output
    
def load_and_predict(file_name, predictor, threshold=0.5):
    """
    load an image, make object detection to an predictor, and visualize detections
    Parameters:
    ----------
    file_name : str
        image file location, in str format
    predictor : sagemaker.predictor.RealTimePredictor
        a predictor loaded from hosted endpoint
    threshold : float
        score threshold for bounding box display
    """
    """
    with open(file_name, 'rb') as image:
        f = image.read()
        b = bytearray(f)"""
    bucket = s3.Bucket('ml-materials')
    img = bucket.Object(f'object_detection_dataset/source_images/{file_name}').get().get('Body').read()
    results = predictor.predict(img)
    #detections = json.loads(results)
    detections = results
    
    fig, detection_filtered = visualize_detection(file_name, detections['prediction'], 
                                                   object_categories, threshold)
    return results, detection_filtered, fig

def only_predict(file_name, predictor, threshold=1.0):
    """
    load an image, make object detection to an predictor, and visualize detections
    Parameters:
    ----------
    file_name : str
        image file location, in str format
    predictor : sagemaker.predictor.RealTimePredictor
        a predictor loaded from hosted endpoint
    threshold : float
        score threshold for bounding box display
    """
    """
    with open(file_name, 'rb') as image:
        f = image.read()
        b = bytearray(f)"""
    bucket = s3.Bucket('ml-materials')
    img = bucket.Object(f'object_detection_dataset/source_images/{file_name}').get().get('Body').read()
    results = predictor.predict(img)
    #detections = json.loads(results)
    detections = results
    
    detection_filtered = []
    for preds in detections['prediction']:
        if float(preds[1])>=threshold:
            detection_filtered.append(preds)
            
    return results, detection_filtered

def verbalize_results(img_fname, result_string,threshold):
    vary = result_string
    timeref = img_fname.split("_")[1].split(".")[0]
    time_in_sec = int(timeref)
    for preds in vary['prediction']:
        if (preds[1]>threshold):
            catname = object_categories[int(preds[0])]
            tlx = int(1920 * preds[2])
            tly = int(1080 * preds[3])
            brx = int(1920 * preds[4])
            bry = int(1080 * preds[5])
            print(f"{catname} {tlx},{tly} {brx},{bry} {preds[1]}")

def predictions_image_sequences(object_detector, image_seq_list, thresh, source_bucket="ml-materials", source_prefix="object_detection_dataset/source_images"):
    na = []
    for times in image_seq_list:
        tsplit = times.split('-')
        tstart = tsplit[0]
        tend = tsplit[1]
        for v in range(int(tstart),int(tend),1):
            fname = f'image_{v:07}.0.png'
            checkfile = s3_objexist(source_bucket,f'{source_prefix}/{fname}')
            if not checkfile:
                fname = f'image_{v:07}.5.png'
                checkfile = s3_objexist(source_bucket,f'{source_prefix}/{fname}')
            print(f"s3://{source_bucket}/{source_prefix}/{fname} {checkfile}")
            results, detection_filtered, f = load_and_predict(fname, object_detector, threshold=thresh)

def predictions_image_list(object_detector, image_list, thresh, source_bucket="ml-materials", source_prefix="object_detection_dataset/source_images"):
    for times in image_list:
        tsplit = times.split('_')[1]
        v = tsplit.split('.')[0]
        fname = f'image_{v}.0.png'
        checkfile = s3_objexist(source_bucket,f'{source_prefix}/{fname}')
        if not checkfile:
            fname = f'image_{v}.5.png'
            checkfile = s3_objexist(source_bucket,f'{source_prefix}/{fname}')
            if not checkfile:
                break
        print(f"s3://{source_bucket}/{source_prefix}/{fname} {checkfile}")
        results, detection_filtered, f = load_and_predict(fname, object_detector, threshold=thresh)

def s3_objexist(bucket,key):
    try:
        s3client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
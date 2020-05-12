import os
from zipfile import ZipFile 
from os import listdir
from os.path import isdir
from PIL import Image
#from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from numpy import load
from numpy import expand_dims
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

import shutil

def extract_face(filename, required_size=(160, 160)):
    	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = os.path.join(directory, filename)
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = os.path.join(directory, subdir)
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
    	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]
 

def trainthedata(filename,modelname):
    print ("train the data is being executed")
    app_root = os.path.dirname(os.path.abspath(__file__))
    app_root=os.path.join(app_root,'images')
    # opening the zip file in READ mode 
    #with ZipFile(os.path.join(app_root, filename), 'r') as zip: 
        # printing all the contents of the zip file 
        #extracting all the files 
        #3print('Extracting all the files now...') 
        #zip.extracptall() 
        #print('Done!') 
    print(os.path.join(app_root,'tour.rar'))
    from pyunpack import Archive
    Archive(os.path.join(app_root,filename)).extractall(app_root)
    filename=filename.split('.')[0]
    os.remove(os.path.join(app_root,'tour.rar'))
    
    #######################################
    #model = load_model('facenet_keras.h5')
    # load train dataset
    #data = SampleModel.parser.parse_args()
    target = os.path.join(app_root, filename)
    traindir=os.path.join(target, 'train')
    testdir=os.path.join(target, 'test')
    print (os.path.join(target, 'train'))
    trainX, trainy = load_dataset(traindir)
    
    print(trainX.shape, trainy.shape)
    print('load test dataset')
    testX, testy = load_dataset(testdir)
    # save arrays to one file in compressed format
    #savez_compressed('faces-cut-tour-dataset.npz', trainX, trainy, testX, testy)
    
    
    ##########################################
    
    #data = load('faces-cut-tour-dataset.npz')
    #trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(target)
    #os.rmdir(target)
    try:
        shutil.rmtree(target)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    # load the facenet model
    print('Loading Model')
    #model = load_model('facenet_keras.h5')
    model = load_model(os.path.join(app_root,'facenet_keras.h5'))
    print('Loaded Model')
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed(os.path.join(app_root,modelname+'encoder.npz'),trainy,testy)
    print('train compleyed')
    
    
    ##################
    
    
    # load dataset
    #data = load('tour-faces-embeddings.npz')
    trainX, trainy, testX, testy = newTrainX, trainy, newTestX, testy
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    #save
    print('save')
    pickle.dump(model, open( os.path.join(app_root,modelname+'.sav'), 'wb'))
    print('saved',modelname)
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
#trainthedata('tour','ootypicsmodel.sav')
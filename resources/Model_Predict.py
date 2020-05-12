from tensorflow.keras.models import load_model
import mtcnn
from PIL import Image 
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
#from matplotlib import pyplot
from numpy import expand_dims
import os
from numpy import load

# get the face embedding for one face
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
    
def predict(filename,modelname):    
    app_root = os.path.dirname(os.path.abspath(__file__))
    app_root=os.path.join(app_root,'images')
    print('# load the model')
    model = load_model(os.path.join(app_root,'facenet_keras.h5'))
    print('loaded the model')
    # load image from file
    print('# load the file')
    image = Image.open(os.path.join(app_root,filename))
    print('# opened the  flie')
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    os.remove(os.path.join(app_root,filename))
    # create the detector, using default weights
    detector = mtcnn.MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    faces = list()
    print('# load the model')
    for i in range(0,len(results)):
        x1, y1, width, height = results[i]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        faces.append(face_array)
        #image.save(str(i)+".jpg")
        #pyplot.subplot(2, 7, i+1)
        #pyplot.axis('off')
        #pyplot.imshow(face_array)
    #pyplot.show()
    facesarray=np.asarray(faces)
    #getembeddings
    newfaceX = list()
    for face_pixels in facesarray:
        embedding = get_embedding(model, face_pixels)
        newfaceX.append(embedding)
    newfaceX = np.asarray(newfaceX)
    #print(newfaceX.shape)
    #normalising
    in_encoder = Normalizer(norm='l2')
    newfaceX = in_encoder.transform(newfaceX)
    loaded_model = pickle.load(open(os.path.join(app_root,modelname+'.sav'), 'rb'))
    aaa=loaded_model.predict(newfaceX)
    data = load(os.path.join(app_root,modelname+'encoder.npz'))
    trainy,testy =  data['arr_0'], data['arr_1']
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    predict_names = out_encoder.inverse_transform(aaa)
    return predict_names
  
    
#predict('file.jpg','ootypicsmodel.sav')
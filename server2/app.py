import os


from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})


vgg16 = tf.keras.models.load_model('./pretrained_models/vgg16_ft_2last_layers.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def count_files(image_dir, image_name):
    count = 0
    for image in os.listdir(image_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(image_dir, image)) and image == image_name:
            count += 1
    return count

""" CLASSIFICATION SECTION
    THIS SECTION HAS ALL THE INITIALIZING CONCERNING THE CLASSIFICATION MODELS
    """
MODELS_PATH = './pretrained_models/'

models_paths = {
    'vgg16': MODELS_PATH + 'vgg16_ft_2last_layers.h5',
    'resnet50': MODELS_PATH + 'ResNet50_ft_2last_layers.h5',
    'inceptionv3': MODELS_PATH + 'Inception_ft_2last_layers.h5',
    'xception': MODELS_PATH + 'Xception_ft_2last_layers.h5',
}

classification_models = {
    'vgg16': tf.keras.models.load_model(models_paths['vgg16']),
    'resnet50': tf.keras.models.load_model(models_paths['resnet50']),
    'inceptionv3': tf.keras.models.load_model(models_paths['inceptionv3']),
    'xception': tf.keras.models.load_model(models_paths['xception']),
}

""" CLUSTERING SECTION
    THIS SECTION HAS ALL THE INITIALIZING CONCERNING THE CLUSTERING MODELS
    """
with open(r'./clustering files/parameters_ft.pkl', 'rb') as f:
    parameters = pickle.load(f)
with open(r'./clustering files/data_preprocessing_ft.pkl', 'rb') as f:
    data_preprocessing = pickle.load(f)
with open(r'./clustering files/DR_models_ft.pkl', 'rb') as f:
    DR_models = pickle.load(f)
with open(r'./clustering files/train_dataset.pkl', 'rb') as f:
    TRAIN_DATASET = pickle.load(f)


clustering_models = {
    'VGG16_FT': tf.keras.models.Model(inputs=classification_models['vgg16'].input, outputs=classification_models['vgg16'].get_layer('global_average_pooling2d').output),
    'InceptionV3_FT':tf.keras.models.Model(inputs=classification_models['inceptionv3'].input, outputs=classification_models['inceptionv3'].get_layer('global_average_pooling2d_1').output),
    'Xception_FT':tf.keras.models.Model(inputs=classification_models['xception'].input, outputs=classification_models['xception'].get_layer('global_average_pooling2d').output),
    'ResNet_FT': tf.keras.models.Model(inputs=classification_models['resnet50'].input,
                                         outputs=classification_models['resnet50'].get_layer(
                                             'global_average_pooling2d_1').output),

}

models_names_mapping = {
    'vgg16': 'VGG16_FT',
    'inceptionv3': 'InceptionV3_FT',
    'xception': 'Xception_FT',
    'resnet50': 'ResNet_FT',
}


""" CLASSIFICATION SECTION
    THIS SECTION HAS ALL THE FUNCTIONS CONCERNING THE CLASSIFICATION MODELS
    """
def classification_process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    return image


def classification_predict_label(image, model):
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    y_pred = model.predict(image)
    labels = ['COVID' if i == 0 else 'Normal' for i in y_pred.argmax(axis=1)]
    percentage = [np.round(y_pred[i][j]*100, 2) for i, j in enumerate(y_pred.argmax(axis=1))]
    y_pred = list(zip(labels, percentage))
    return y_pred


@app.route('/image', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"message": "No image found to upload!"}), 404

    image = request.files['image']
    patient_name = request.form.get('patient_name')

    if image.filename == '':
        return jsonify({"message": "No file name was found!"}), 404

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        extension = os.path.splitext(filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], patient_name + extension)
        image.save(image_path)
        return jsonify({
            "image": patient_name + extension,
            "patient_name" : patient_name,
            "extension" : extension
        }), 200


@app.route('/classification/predict', methods=['POST'])
def classification_prediction():
    # image preprocessing
    image_path = os.path.join(app.config['UPLOAD_FOLDER']) + '/' + request.get_json()['image']
    image = classification_process_image(image_path)
    results = {}
    for model in request.get_json()['models']:
        results[model] = classification_predict_label(image, classification_models[model])
    return jsonify(results)


""" CLUSTERING SECTION
    THIS SECTION HAS ALL THE FUNCTIONS CONCERNING THE CLUSTERING MODELS
    """
def Normalize(data):
    norm_images = data/255
    # Expand dimension of grayscale images
    if norm_images.ndim == 3:
        norm_images = np.expand_dims(norm_images,axis=3)
    return norm_images

def Image_features(image,model):
    # Create 3 channels for grayscale images
    if image.shape[2] == 1:
        image = image.repeat(3, axis=2)
    # Expand image dimension to become (1,224,224,3)
    image= np.expand_dims(image,axis=0)
    image_features = model.predict(image)
    return np.array(image_features)


def Extract_features_TL_models_from_path(images, models):
    all_features = {}

    # Loop over each model
    print('#### Start Features Extraction ####')
    for m_name, model in models.items():
        print('Start with model {}'.format(m_name))
        image_features = []
        # Loop over each image in the dataset
        for image in images:
            # Load image with its label
            image = cv2.imread(image)
            # Resize image
            img = cv2.resize(image, model.input_shape[1:3])
            # Normalize image
            img_norm = Normalize(img)
            # Extract features
            feature_map = Image_features(img_norm, model)
            image_features.append(feature_map)

        image_features = np.array(image_features, dtype=np.double)
        image_features = image_features.squeeze(axis=1)
        all_features[m_name] = image_features
    print('#### End Features Extraction ####')

    return all_features

def labels_to_clusters (labels,clusters):
    normal = 0
    covid = 0
    # Count the number of covid and normal cases in cluster 0 to determine which label is the majority
    for cluster,label in zip(clusters,labels):
        if (cluster == 0 and label =='COVID'):
            covid +=1
        elif (cluster == 0 and label =='NORMAL'):
            normal +=1
    if covid > normal :
        labels = [0 if l=='COVID' else 1 for l in labels]
        ticks = ['COVID','NORMAL']
    else:
        labels = [1 if l=='COVID' else 0 for l in labels]
        ticks = ['NORMAL','COVID']
    return labels

def get_cluster_label (labels,clusters):
    normal = 0
    covid = 0
    # Count the number of covid and normal cases in cluster 0 to determine which label is the majority
    for cluster,label in zip(clusters,labels):
        if (cluster == 0 and label =='COVID'):
            covid +=1
        elif (cluster == 0 and label =='NORMAL'):
            normal +=1
    if covid > normal :
        labels = [0 if l=='COVID' else 1 for l in labels]
        labels = ['COVID','NORMAL']
    else:
        labels = [1 if l=='COVID' else 0 for l in labels]
        labels = ['NORMAL','COVID']
    return labels


def Scaling_data(data):
    scaler = StandardScaler()
    STD_DF = scaler.fit_transform(data)
    return STD_DF

def predict(features_map,parameters,DR_models,TL_model,clustering_algo,dr_algo=None):
    # Get the desired model to make clustering
    keys = [key for key in parameters[TL_model]['Models'].keys() if key.startswith(clustering_algo)]
    if dr_algo == 'Only':
        model = parameters[TL_model]['Models'][keys[0]]
        try:
            y_pred = model.predict(features_map[TL_model].reshape(1,-1))
        except:
            y_pred = model.predict(features_map[TL_model])
        return y_pred
    elif dr_algo =='PCA':
        model = parameters[TL_model]['Models'][keys[1]]
        reducer = DR_models[TL_model][0]
        PCA_DF = reducer.transform(features_map[TL_model])
        try:
            y_pred = model.predict(PCA_DF.reshape(1,-1))
        except:
            y_pred = model.predict(PCA_DF)
        return y_pred
    else:
        model = parameters[TL_model]['Models'][keys[2]]
        reducer = DR_models[TL_model][1]
        UMAP_DF = reducer.transform(features_map[TL_model])
        try:
            y_pred = model.predict(UMAP_DF.reshape(1,-1))
        except:
            y_pred = model.predict(UMAP_DF)
        return y_pred








@app.route('/clustering/predict', methods=['POST'])
def clustering_prediction():
    results = {}

    image_path = os.path.join(app.config['UPLOAD_FOLDER']) + '/' + request.get_json()['image']
    features_map = Extract_features_TL_models_from_path(images=[image_path], models=clustering_models)


    for model in request.get_json()['models']:
        results[model] = {}
        for dr_method in request.get_json()['dim_reduction_methods']:
            Cl_DR_algo = request.get_json()['algorithm'] + '/' + dr_method
            y_pred = predict(features_map, parameters, DR_models,
                             TL_model=models_names_mapping[model] ,
                             clustering_algo=request.get_json()['algorithm'],
                             dr_algo=dr_method)
            label = get_cluster_label(labels=TRAIN_DATASET['Labels'],
                                      clusters=parameters[models_names_mapping[model]²]['Clusters'][Cl_DR_algo])
            results[model][dr_method] = label[y_pred[0]]
    return jsonify(results)


if __name__ == '__main__':
    app.run()

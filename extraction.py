from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras.models import Model
import numpy as np

# extract VGG features for each image's encoding
def extract_features(directory):

    model = VGG16(weights='imagenet')
    models = []
    for i in range(3,7):
        if i == 4:
            continue
        new_input = model.input
        hidden_layer = model.layers[-i].output
        model_new = Model(new_input, hidden_layer)
        models.append(model_new)
        print('Last {} layer out VGG\n'.format(i),model_new.summary())
        
    # container, in form of [dict of last_3_layers_drop_out,dict of last_4_layers_drop_out,dict of last_5_layers_drop_out]
    features = [dict(),dict(),dict()]
    for i,model in enumerate(models):
        for name in listdir(directory):
            filename = directory + '/' + name
            image = load_img(filename, target_size=(224, 224))
            image = np.array([img_to_array(image)])
            image = preprocess_input(image)
            feature = model.predict(image, verbose=1)
            print(feature.shape)
            image_id = name.split('.')[0]
            print(image_id)
            features[i][image_id] = feature
    return features

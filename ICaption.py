import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os


image_model = InceptionV3(weights='imagenet')
image_model = Model(image_model.input, image_model.layers[-2].output)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


model = load_model('image_captioning_model.h5')

# Generate caption for an image
def generate_caption(image_path, max_length=34):
    image = preprocess_image(image_path)
    image_features = image_model.predict(image, verbose=0)
    
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
    return in_text.split(' ')[1:-1]


def backup(model, tokenizer):
    model.save('image_captioning_model_backup.h5')
    
    with open('tokenizer_backup.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    

def optimize(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    


image_path = 'example.jpg'
caption = generate_caption(image_path)
print('Generated Caption:', ' '.join(caption))


backup(model, tokenizer)


optimize(model)
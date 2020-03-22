from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from keras.models import load_model

img = img_to_array(load_img(""))
model = load_model('my_model.h5')
output = model.predict(img)
save_img('out.jpg', output)

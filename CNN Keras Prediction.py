from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

test_image = image.load_img('testImage.png')
test_image_with_noise = image.load_img('testImageWithNoise.png')

plt.imshow(test_image)
plt.show()

test_image = test_image.resize((28,28))
test_image = image.img_to_array(test_image)
test_image = test_image.astype('float32') / 255.
test_image = np.delete(test_image, [0] ,axis=2)
test_image = np.delete(test_image, [0] ,axis=2)
test_image = np.reshape(test_image, (1, 28, 28, 1))
print(test_image.shape)

plt.imshow(test_image_with_noise)
plt.show()

test_image_with_noise = test_image_with_noise.resize((28,28))
test_image_with_noise = image.img_to_array(test_image_with_noise)
test_image_with_noise = test_image_with_noise.astype('float32') / 255.
test_image_with_noise = np.delete(test_image_with_noise, [0] ,axis=2)
test_image_with_noise = np.delete(test_image_with_noise, [0] ,axis=2)
test_image_with_noise = np.reshape(test_image_with_noise, (1, 28, 28, 1))
print(test_image_with_noise.shape)

model = load_model('my_model.h5')
prediction = model.predict(test_image)
predictionNoise = model.predict(test_image_with_noise)

result = np.where(prediction[0] == prediction[0].max())[0][0]
print("Prediction for given image - ", result)
result = np.where(predictionNoise[0] == predictionNoise[0].max())[0][0]
print("Prediction for given image with noise - ", result)
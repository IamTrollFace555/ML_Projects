from PIL import Image
import numpy as np


def preprocess_image():
    # Open image saved by board.py
    image = Image.open('user_image.jpg')

    # Resize the image
    new_image = image.resize((28, 28))
    new_image.save('small_image.jpg')

    # Open the resized image
    image2 = Image.open('small_image.jpg')

    # convert the image to a list
    np_img = np.array(image2.getdata()).reshape(28, 28, 3)
    temp = np_img.tolist()

    # To make the pixels brighter
    extra_brightness = 40

    # To remove imperfections produced by the image resizing
    threshold = 30 + extra_brightness

    # Convert the image to a numpy array and apply brightness and correction changes
    temp = [min(255, x[0] + extra_brightness) if x[0] > threshold else 0 for y in temp for x in y]
    np_img = np.array(temp).reshape(28, 28)

    return np_img


if __name__ == "__main__":
    img = preprocess_image()
    print(img.reshape(1, 784).tolist()[0])

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


class ImageSegmenter():
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights('usnet256acnn_weights__92-86.h5')

    def __call__(self, image_bytes):
        image = tf.image.decode_png(image_bytes)
        image_size = tf.shape(image)
        image = self.preprocess(image)
        mask = self.model.predict(image)
        mask = self.mask2image(mask)
        mask = tf.image.resize(mask, (image_size[0], image_size[1]))
        mask = tf.floor(mask)
        return tf.squeeze(mask)/3

    def preprocess(self, image):
        image = tf.image.resize(image, (256, 256), antialias=True)
        if image.shape.ndims == 2 :
            image = image[:, :, tf.newaxis]
        image = tf.repeat(image, repeats=3,axis=-1)
        image_min = tf.math.reduce_min(image)
        image_max = tf.math.reduce_max(image)
        image = (image - image_min)/(image_max - image_min)
        image = image[tf.newaxis, :, :, :]
        return image


    def build_model(self, aug_layer = tf.keras.layers.GaussianNoise(1)):
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=[256, 256, 3], include_top=False, weights=None)


        layer_names = [
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3',
            'block5_conv3'
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = True

        with tf.device('/GPU:0'):
            up_stack = [
                pix2pix.upsample(128, 3),  # 16x16 -> 32x32
                pix2pix.upsample(64, 3),   # 32x32 -> 64x64
                pix2pix.upsample(32, 3),   # 32x32 -> 64x64
            ]
        
        with tf.device('/GPU:0'):
            inputs = tf.keras.layers.Input(shape=[256, 256, 3])
            aug = aug_layer(inputs)

            # Downsampling through the model
            skips = down_stack(aug)
            x = skips[-1]
            skips = reversed(skips[:-1])

            # Upsampling and establishing the skip connections
            for up, skip in zip(up_stack, skips):
                x = up(x)
                concat = tf.keras.layers.Concatenate()
                x = concat([x, skip])

            # This is the last layer of the model
            last = tf.keras.layers.Conv2DTranspose(
                filters=4, kernel_size=3, strides=2,
                padding='same')  #64x64 -> 128x128

            x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def mask2image(self, mask):
        mask = tf.argmax(mask, axis=-1)
        mask = mask[:, :, :, tf.newaxis]
        return mask
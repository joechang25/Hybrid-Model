from . import correct_pad
from . import get_submodules_from_kwargs
from . import imagenet_utils
from tensorflow.keras import activations

backend = None
layers = None
models = None
keras_utils = None


def CustomNet(weights=None, input_shape=None, include_top=False, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    img_input = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', name='Conv')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm2')(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv22')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm22')(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D()(x)

    #x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv3')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm3')(x)
    #x = layers.Activation(activations.relu)(x)

    x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv33')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm33')(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D()(x)

    #x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv4')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm4')(x)
    #x = layers.Activation(activations.relu)(x)

    #x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv44')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm44')(x)
    #x = layers.Activation(activations.relu)(x)
    #x = layers.MaxPooling2D()(x)
    

    #x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv5')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm5')(x)
    #x = layers.Activation(activations.relu)(x)
  
    #x = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', name='Conv55')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm55')(x)
    #x = layers.Activation(activations.relu)(x)
    #x = layers.MaxPooling2D()(x)
  
    #x = layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='Conv6')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm6')(x)
    #x = layers.Activation(activations.relu)(x)
    
    #x = layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='Conv66')(x)
    #x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm66')(x)
    #x = layers.Activation(activations.relu)(x)

    x = layers.Flatten()(x)
    #x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(50)(x)
    #x = layers.Dense(100)(x)
    #x = layers.Dense(10)(x)
    model = models.Model(img_input, x, name='Custom Model')

    return model

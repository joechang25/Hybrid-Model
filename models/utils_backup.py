import tensorflow as tf
import models


_KERAS_BACKEND = tf.keras.backend
_KERAS_LAYERS = tf.keras.layers
_KERAS_MODELS = tf.keras.models
_KERAS_UTILS = tf.keras.utils

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

    
def get_normalize_fn(arch):
    return {
        'xception': models.xception.preprocess_input,
        'vgg16': models.vgg16.preprocess_input,
        'vgg19': models.vgg19.preprocess_input,
        'resnet18': models.resnet.preprocess_input,
        'resnet34': models.resnet.preprocess_input,
        'resnet50': models.resnet.preprocess_input,
        'resnet101': models.resnet.preprocess_input,
        'resnet152': models.resnet.preprocess_input,
        'resnet50_v2': models.resnet_v2.preprocess_input,
        'resnet101_v2': models.resnet_v2.preprocess_input,
        'resnet152_v2': models.resnet_v2.preprocess_input,
        'resnext50': models.resnext.preprocess_input,
        'resnext101': models.resnext.preprocess_input,
        'inception_v3': models.inception_v3.preprocess_input,
        'inception_resnet_v2': models.inception_resnet_v2.preprocess_input,
        'mobilenet': models.mobilenet.preprocess_input,
        'mobilenet_v2': models.mobilenet_v2.preprocess_input,
        'mobilenet_v3_small': models.mobilenet_v3.preprocess_input,
        'mobilenet_v3_large': models.mobilenet_v3.preprocess_input,
        'densenet121': models.densenet.preprocess_input,
        'densenet169': models.densenet.preprocess_input,
        'densenet201': models.densenet.preprocess_input,
        'nasnet_mobile': models.nasnet.preprocess_input,
        'nasnet_large': models.nasnet.preprocess_input,
        'efficientnet_b0': models.efficientnet.preprocess_input,
        'efficientnet_b1': models.efficientnet.preprocess_input,
        'efficientnet_b2': models.efficientnet.preprocess_input,
        'efficientnet_b3': models.efficientnet.preprocess_input,
        'efficientnet_b4': models.efficientnet.preprocess_input,
        'efficientnet_b5': models.efficientnet.preprocess_input,
        'efficientnet_b6': models.efficientnet.preprocess_input,
        'efficientnet_b7': models.efficientnet.preprocess_input,
    }[arch]


def get_resumed_model(args):
    print("  => loading checkpoint '{}'".format(args.resume))
    custom_ob = {'hard_swish' : models.mobilenet_v3.hard_swish, 'tf' : tf}
    return tf.keras.models.load_model(args.resume, compile=False, custom_objects= custom_ob)


def check_input_shape(args):
    maps = {'xception': 299,
            'resnet50_v2': 299,
            'resnet101_v2': 299,
            'resnet152_v2': 299,
            'inception_v3': 299,
            'inception_resnet_v2': 299,
            'nasnet_large': 331,
            'efficientnet_b1': 240,
            'efficientnet_b2': 260,
            'efficientnet_b3': 300,
            'efficientnet_b4': 380,
            'efficientnet_b5': 456,
            'efficientnet_b6': 528,
            'efficientnet_b7': 600,
    }
    warning = "     warning: input size '{}' != default value '{}'"
    if args.arch in maps and args.image_size != maps[args.arch]:
        print(warning.format(args.image_size, maps[args.arch]))


def get_pretrained_model(args):
    print('  => using pre-trained model')
    model_class = models.model_class[args.arch]
    input_shape = (args.image_size, args.image_size, 3)

    check_input_shape(args)
    return model_class(weights='imagenet',
                      input_shape=input_shape,
                      include_top=True)


def add_model_top(model, args):
    dropout = args.dropout

    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    if dropout > 0.0 and dropout < 1.0:
        x = tf.keras.layers.Dropout(dropout)(x)
    if args.mixed:
        x = tf.keras.layers.Dense(1)(x)
    else:
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(20, activation='relu')(x)
        x = tf.keras.layers.Dense(1, dtype='float32')(x)
    return x


def get_custom_model(args):
    model_class = models.model_class[args.arch]
    input_shape = (args.image_size, args.image_size, 3)

    model = model_class(weights=None,
                        input_shape=input_shape,
                        include_top=False)
    output = add_model_top(model, args)
    model = tf.keras.models.Model(name=args.arch,
                                  inputs=model.input,
                                  outputs=output)

    if args.torch_init:
        print("  => using 'torch' layer initialization")
        kernel_initializer = tf.keras.initializers.he_uniform()
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.1
                layer.epsilon = 1e-3 if args.mixed else 1e-5
            elif hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = kernel_initializer
    else:
        print("  => using 'keras' layer initialization")

    return model


def get_model(args):
    print("  => creating model '{}'".format(args.arch))

    if args.resume:
        model = get_resumed_model(args)
    elif args.resize:
        model = get_pretrained_model(args)
    elif args.pretrained:
        model = get_pretrained_model(args)
        output = add_model_top(model, args)
        model = tf.keras.models.Model(name=args.arch,
                                      inputs=model.input,
                                      outputs=output)

        if args.torch_init:
            print("  => using 'torch' layer initialization")
            kernel_initializer = tf.keras.initializers.he_uniform()
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.momentum = 0.1
                    layer.epsilon = 1e-3 if args.mixed else 1e-5
                elif hasattr(layer, 'kernel_initializer'):
                    layer.kernel_initializer = kernel_initializer
            else:
                print("  => using 'keras' layer initialization")
    else:
        model = get_custom_model(args)

    for layer in model.layers:
        layer.trainable = True

    return model

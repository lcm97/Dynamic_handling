from keras.models import Model
from keras.layers import Conv2D, Input, Flatten, Dense, MaxPooling1D, TimeDistributed, Lambda
from keras.layers.merge import concatenate
import tensorflow as tf
import numpy as np


def lambda_fun(x, num_split, index):
    split = tf.split(x, num_split, axis=1)
    split_index = split[index]
    return split_index


def spatial_model():
    image = Input(shape=(12, 16, 3))

    layer = Conv2D(256, (5, 5))(image)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Flatten()(layer)

    layer = Dense(256, activation='relu')(layer)

    return Model(image, layer)


def temporal_model():
    image = Input(shape=(12, 16, 20))

    layer = Conv2D(256, (5, 5))(image)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)

    return Model(image, layer)


def maxpoolings():
    inter_represent = Input(shape=(16, 256))
    """level 1"""
    layer1 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(inter_represent)
    """level 2"""
    sub_layer1 = Lambda(lambda_fun, arguments={'num_split': 2, 'index': 0})(inter_represent)
    sub_layer2 = Lambda(lambda_fun, arguments={'num_split': 2, 'index': 1})(inter_represent)
    layer2 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer1)
    layer3 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer2)
    """level 3"""
    sub_layer3 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 0})(inter_represent)
    sub_layer4 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 1})(inter_represent)
    sub_layer5 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 2})(inter_represent)
    sub_layer6 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 3})(inter_represent)
    layer4 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer3)
    layer5 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer4)
    layer6 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer5)
    layer7 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer6)
    """level 4"""
    sub_layer7 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 0})(inter_represent)
    sub_layer8 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 1})(inter_represent)
    sub_layer9 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 2})(inter_represent)
    sub_layer10 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 3})(inter_represent)
    sub_layer11 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 4})(inter_represent)
    sub_layer12 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 5})(inter_represent)
    sub_layer13 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 6})(inter_represent)
    sub_layer14 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 7})(inter_represent)
    layer8 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer7)
    layer9 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer8)
    layer10 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer9)
    layer11 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer10)
    layer12 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer11)
    layer13 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer12)
    layer14 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer13)
    layer15 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer14)
    layer = concatenate([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8,
                         layer9, layer10, layer11, layer12, layer13, layer14, layer15], 1)
    """the out put shape is (15,256),for each maxpooling layer generate (1,256)"""
    return Model(inter_represent, layer)


def spatial_model_multi():
    """Input with 16x(12,16,3) frames with time distribute"""
    frames = Input(shape=(16, 12, 16, 3))
    layer = TimeDistributed(spatial_model())(frames)
    """The output shape is (16,256)"""
    return Model(frames, layer)


def temporal_model_multi():
    optical_flows = Input(shape=(16, 12, 16, 20))
    layer = TimeDistributed(temporal_model())(optical_flows)
    return Model(optical_flows, layer)


def temporal_pyramid_concate():
    spatial_input = Input(shape=(1, 15, 256))
    temporal_input = Input(shape=(1, 15, 256))

    """Now,the output shape is (2,15,256)"""
    layer = concatenate([spatial_input, temporal_input], 1)
    return Model([spatial_input, temporal_input], layer)


def layer_merge():
    X1 = Input(shape=(4096,))
    X2 = Input(shape=(4096,))
    layer = concatenate([X1, X2], 1)
    return Model([X1, X2], layer)


def fc_1():
    layer_input = Input(shape=(2, 15, 256))
    layer = Flatten()(layer_input)
    layer = Dense(4096, activation='relu')(layer)
    return Model(layer_input, layer)


def fc_23():
    layer_input = Input(shape=(8192,))
    layer = Dense(8192, activation='relu')(layer_input)
    layer = Dense(51, activation='softmax')(layer)
    return Model(layer_input, layer)


def forward(req_input, req_next, node):
    bytestr = req_input

    if req_next == 'spatial':

        X = np.fromstring(bytestr, np.uint8).reshape(16, 12, 16, 3)
        node.model = spatial_model_multi() if node.model is None else node.model
        output = node.model.predict(np.array([X]))
        name = 'block1'
        node.log('finish spatial forward')
        return output, name

    elif req_next == 'temporal':
        node.log('temporal gets data')
        X = np.fromstring(bytestr, np.uint8).reshape(16, 12, 16, 20)
        node.model = temporal_model_multi() if node.model is None else node.model
        output = node.model.predict(np.array([X]))
        name = 'block'
        node.log('finish temporal forward')
        return output, name

    elif req_next == 'block1':
        node.log('block1 gets data')
        X = np.fromstring(bytestr, np.float32).reshape(16, 256)
        node.input.append(X)
        node.log('input size', str(len(node.input)))
        # if the size is not enough, store in the queue and return.
        if len(node.input) < 2:
            node.release_lock()
            return
        # too many data packets, then drop some data.
        while len(node.input) > 2:
            node.input.popleft()

        mp_model = maxpoolings()
        output1 = mp_model.predict(np.array([node.input[0]]))
        output2 = mp_model.predict(np.array([node.input[1]]))

        con_model = temporal_pyramid_concate()
        X = con_model.predict([np.array([output1]), np.array([output2])])

        inter_dense = fc_1()
        output = inter_dense.predict(X)
        name = 'block2'
        node.log('fc_1 model inference')
        return output, name

    elif req_next == 'block2':
        node.log('block2 gets data')
        X = np.fromstring(bytestr, np.float32).reshape(4096,)
        node.input.append(X)
        node.log('input size', str(len(node.input)))
        # if the size is not enough, store in the queue and return.
        if len(node.input) < 2:
            node.release_lock()
            return
        # too many data packets, then drop some data.
        while len(node.input) > 2:
            node.input.popleft()

        merge = layer_merge()
        X = merge.predict([np.array([node.input[0]]), np.array([node.input[1]])])
        final_dense = fc_23()
        output = final_dense.predict(X)
        name = 'initial'
        node.log('finish fc_23 forward')
        return output, name


# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.oct import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

import datetime
from flask import render_template, redirect, request, url_for, flash
from werkzeug.utils import secure_filename
from flask_login import (
    current_user,
    login_user,
    logout_user
)



import os
import random

import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.utils import load_img, array_to_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Input, Activation, SeparableConv2D, BatchNormalization, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

from skimage.io import imshow

import matplotlib.pyplot as plt 
import cv2

from pathlib import Path
import random
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from tqdm import tqdm


from clickhouse_driver import Client


import boto3
from botocore.client import Config

from werkzeug.utils import secure_filename

from botocore.client import Config

s3 = boto3.resource('s3',
                    endpoint_url='http://10.10.10.101:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    )

client = Client('10.10.10.81',
                user='kirill',
                password='',
                secure=False,
                verify=False,
                database='firstdb',
                compression=True,
                settings={'use_numpy': True})

connection = dict(database='firstdb',
                  host='http://10.10.10.81:8123',
                  user='kirill',
                  password='')

IMG_WIDTH_HEIGHT = 256
IMG_CHANNELS = 1
classes = 4

def convolutional_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters,
                    kernel_size = 3,
                    activation='relu',
                    padding='same',
                    kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    conv = Conv2D(n_filters, 
                    kernel_size = 3, 
                    activation='relu',
                    padding='same',
                    kernel_initializer=tf.keras.initializers.HeNormal())(conv)
    if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
    if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2,2))(conv)
    else:
            next_layer = conv
        #conv = BatchNormalization()(conv)
    skip_connection = conv
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):

        up = Conv2DTranspose(
                    n_filters,
                    kernel_size = 3,
                    strides=(2,2),
                    padding='same')(expansive_input)
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters,
                    kernel_size = 3,
                    activation='relu',
                    padding='same',
                    kernel_initializer=tf.keras.initializers.HeNormal())(merge)
        conv = Conv2D(n_filters,  
                    kernel_size = 3,  
                    activation='relu',
                    padding='same',
                    kernel_initializer=tf.keras.initializers.HeNormal())(conv)
        
        return conv
    
def unet_model(input_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT
                           , IMG_CHANNELS), n_filters=32, n_classes=classes):
            with tf.device('/GPU:1'):  
                inputs = Input(input_size)

                #contracting path
                cblock1 = convolutional_block(inputs, n_filters)
                cblock2 = convolutional_block(cblock1[0], 2*n_filters)
                cblock3 = convolutional_block(cblock2[0], 4*n_filters)
                cblock4 = convolutional_block(cblock3[0], 8*n_filters, dropout_prob=0.2) 
                cblock5 = convolutional_block(cblock4[0],16*n_filters, dropout_prob=0.2, max_pooling=None)     
                #expanding path
                ublock6 = upsampling_block(cblock5[0], cblock4[1],  8 * n_filters)
                ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
                ublock8 = upsampling_block(ublock7,cblock2[1] , n_filters*2)
                ublock9 = upsampling_block(ublock8,cblock1[1],  n_filters)
                conv9 = Conv2D(n_classes,
                            1,
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(ublock9)
                #conv10 = Conv2D(n_classes, kernel_size=1, padding='same', activation = 'softmax')(conv9) 
                conv10 = Activation('softmax')(conv9)
                model = tf.keras.Model(inputs=inputs, outputs=conv10)
                return model


@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/uploaderoct', methods = ['GET', 'POST'])





def upload_file2():
    
    if request.method == 'POST':
        f = request.files['file']
        
        #f2=f
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        f.filename = suffix + f.filename
        
        #f2.filename = 'seg' + suffix + f.filename
        f.save(secure_filename(f.filename))       
    
    s3_r = boto3.resource('s3',
                    endpoint_url='http://10.10.10.101:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    )
      
      
      # отправка файла в minio
    s3_r.Bucket('data').upload_file(f"{f.filename}", f"{f.filename}")
      #file = s3.Bucket('data').download_file(f"{f.filename}", f"{f.filename}")
      #csv_file = csv.reader(file)
      #client.execute("INSERT INTO iris FORMAT CSV", csv_file)
      
    s3 = boto3.client('s3',
                    endpoint_url='http://10.10.10.101:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',)
      
    bucket='data'
    key = f.filename
    
    result = s3.list_objects(Bucket = bucket, Prefix='/')
    for o in result.get('Contents'):
                data = s3.get_object(Bucket=bucket, Key=f.filename)
                body = data['Body']
    bucket = s3_r.Bucket(bucket)
    
    
    img = bucket.Object(key).get().get('Body').read()
    nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_GRAYSCALE)
            
    #im2=cv2.imread(nparray, cv2.IMREAD_GRAYSCALE)
    nparray=cv2.resize(nparray, (256,256)) # resize to 180,180 as that is on which model is trained on
    print(nparray.shape)
    img2 = tf.expand_dims(nparray, 0)
    img2 = tf.expand_dims(img2, -1)    
    model = tf.keras.models.load_model('./apps/oct/saved_model/my_model')
    
    model.load_weights('./apps/oct/checkpoints/my_checkpoint')
    
    predictions = model.predict(img2)
    
    lable1 = predictions[0,:,:,1]
    lable1.shape

    # Load images as greyscale but make main RGB so we can annotate in colour
    lable1  = np.squeeze(predictions[0,:,:,1])
    lable2  = np.squeeze(predictions[0,:,:,2])
    lable3  = np.squeeze(predictions[0,:,:,3])
    seg = np.zeros((256,256))
    seg[lable1 > 0.95] = 1
    seg[lable2 > 0.95] = 2
    seg[lable3 > 0.95] = 3
    main = np.squeeze(img2[0])
    
    from skimage.measure import label
    
    #import self
    #color = self.color
    import skimage
    
    img_data = plt.imshow(skimage.color.label2rgb(seg,main,colors=[(255,100,0),(0,0,255),(0,123,100)],alpha=0.1, bg_label=0, bg_color=None))
    #plt.show()
    import io
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    plt.savefig('./apps/static/assets/img/foo.png')
    img_data.seek(0)
    #f2 = img_data
    #f2.filename = "seg" + suffix + f.filename
    #plt.savefig(img_data, format='png')
    #img_data.save(f.filename)
    
    #s3.upload_fileobj(img_data,'data',key)
    
    
    s3_r.Object('data',f"{f.filename}.png").put(Body=img_data.getvalue(),
                                          ContentType='image/png') 
    
    #результат вызов фукции
    source= '''test()'''
    
    
    
      
    return render_template('home/octseg.html', url = '/static/assets/img/foo.png', url2 = f'/static/assets/img/{f.filename}', source=source)


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
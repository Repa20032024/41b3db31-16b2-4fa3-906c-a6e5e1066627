# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import csv
import datetime
from flask import render_template, redirect, request, url_for, flash
from werkzeug.utils import secure_filename
from flask_login import (
    current_user,
    login_user,
    logout_user
)


from clickhouse_driver import Client
from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users

import boto3
from botocore.client import Config
import csv
import pandas as pd
from io import StringIO
from werkzeug.utils import secure_filename
from flask import render_template, redirect, request, url_for, flash
import pandahouse as ph

from botocore.client import Config

s3 = boto3.resource('s3',
                    endpoint_url='http://10.10.10.30:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    )

client = Client('10.10.10.81',
                user='kirill',
                password='tammenyanet88DL',
                secure=False,
                verify=False,
                database='firstdb',
                compression=True,
                settings={'use_numpy': True})

connection = dict(database='firstdb',
                  host='http://10.10.10.81:8123',
                  user='kirill',
                  password='tammenyanet88DL')



from apps.authentication.util import verify_pass

@blueprint.route('/')
def route_default():
    return redirect(url_for('authentication_blueprint.login'))

# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']
       
        # Locate user
        user = Users.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):

            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template('accounts/login.html',
                               msg='Wrong user or password',
                               form=login_form)

    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']
        fullname = request.form['fullname']
        
       
        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Такой пользователь уже существует',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email уже зарегистрирован',
                                   success=False,
                                   form=create_account_form)

        
        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        # Delete user from session
        logout_user()

        return render_template('accounts/register.html',
                               msg='Пользователь успешно создан.',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)

@blueprint.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
   if request.method == 'POST':
      f = request.files['file']
      suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      f.filename = suffix + f.filename
      f.save(secure_filename(f.filename))
      
      s3 = boto3.resource('s3',
                    endpoint_url='http://10.10.10.30:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    )
      
      
      # отправка файла в minio
      s3.Bucket('data').upload_file(f"{f.filename}", f"{f.filename}")
      #file = s3.Bucket('data').download_file(f"{f.filename}", f"{f.filename}")
      #csv_file = csv.reader(file)
      #client.execute("INSERT INTO iris FORMAT CSV", csv_file)
      
      s3 = boto3.client('s3',
                    endpoint_url='http://10.10.10.30:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',)
      
      bucket='data'
      result = s3.list_objects(Bucket = bucket, Prefix='/')
      for o in result.get('Contents'):
        data = s3.get_object(Bucket=bucket, Key=f.filename)
        body = data['Body']
        csv_string = body.read().decode('utf-8')
      df = pd.read_table(StringIO(csv_string), sep=';', encoding='utf-16', dtype={'Pat-ID:':'object'})
      
      df.columns=df.columns.str.replace(':','')
      df.columns=df.columns.str.replace(':','')
      df.replace(',','.', regex=True,inplace=True)
      df = df.drop(['Last Name','First Name','D.o.Birth'], axis=1)
      df.dropna(subset = ['Pat-ID'], inplace = True)
      
      columnName = list(df.columns.values)
      
      def getColumnDtypes(dataTypes):
        dataList = []
        for x in dataTypes:
            if(x == 'int64'):
                dataList.append('int')
            elif (x == 'float64'):
                dataList.append('float')
            elif (x == 'bool'):
                dataList.append('boolean')
            else:
                dataList.append('varchar')
        return dataList
      
      
      columnDataType = getColumnDtypes(df.dtypes)
      
      createTableStatement = 'CREATE TABLE IF NOT EXISTS pentacam ('
      for i in range(len(columnDataType)):
            createTableStatement = createTableStatement + ' ' + "`" + columnName[i] + "`" + ' ' + columnDataType[i] + ','
      createTableStatement = createTableStatement[:-1] + ' )' +' '+'ENGINE = MergeTree()' + ' ' + 'PRIMARY KEY (`Pat-ID`,`Exam Date`,`Exam Time`)'
      
      client.execute(createTableStatement)
      
      client.insert_dataframe(f'INSERT INTO pentacam VALUES', df)
      
      #результат вызов фукции
      source= '''test()'''
      
      
   return render_template('home/database-ingest-data.html', source=source)





@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login')) 

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500

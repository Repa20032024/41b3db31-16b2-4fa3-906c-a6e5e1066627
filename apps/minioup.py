
from minio import Minio
from minio.error import InvalidResponseError
from io import BytesIO
host = "10.10.10.30:9000"
access_key = "minioadmin"
secret_key = "minioadmin"
minioClient = Minio(host, access_key=access_key, secret_key=secret_key, secure=False)

text = "My minio content"
bucket = "data"
content = BytesIO(bytes(text,'utf-8'))
key = 'sample.text'
size = content.getbuffer().nbytes

try:    
    minioClient.put_object(bucket,key,content,size)
    print("done!")
except InvalidResponseError as err:
    print("error: " + err)
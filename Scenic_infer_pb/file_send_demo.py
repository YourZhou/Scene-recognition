import requests

upload_url='http://127.0.0.1/infer'

files = {'img':open('D:\\rock554.jpg','rb')}
upload_res=requests.post(upload_url,files=files)
print(upload_res.text)


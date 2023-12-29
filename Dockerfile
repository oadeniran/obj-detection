FROM python:3.9

WORKDIR /code

EXPOSE 5000

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt


CMD [ "python3", "app.py" ]
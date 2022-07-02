
    
FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "../model_api/uber_fare_price_regressor_api.py"]    

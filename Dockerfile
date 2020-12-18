FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire scikit-learn==0.20.4 pandas==0.24.2 kfp==0.2.5
RUN pip install google-cloud-storage
COPY ./august-sandbox-298320-580249f0836f.json /august-sandbox-298320-580249f0836f.json
RUN pip install xgboost

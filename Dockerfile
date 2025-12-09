#tf GPU base image to
#FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest
FROM python:3.10-slim-bullseye

#working dir inside container
WORKDIR /app

#copy, install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy project files
COPY config.py .
COPY data.py .

#create directories for outputs
RUN mkdir -p /app/data /app/models /app/logs /app/results /tfrecords

#environment variable to avoid matplotlib display issues
ENV MPLBACKEND=Agg

#run data when container starts
CMD ["python", "run.py"]
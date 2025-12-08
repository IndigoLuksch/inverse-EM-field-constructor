#tf GPU base image to
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest

#working dir inside container
WORKDIR /app

#copy, install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy project files
COPY config.py .
COPY data_generation.py .

#create directories for outputs
RUN mkdir -p /app/data /app/models /app/logs /app/results

#environment variable to avoid matplotlib display issues
ENV MPLBACKEND=Agg

#run data_generation.py when container starts
CMD ["python", "data_generation.py"]
FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scripts (the CSV is intentionally NOT copied in -
# it's mounted at runtime so it never ends up in the image or on GitHub)
COPY data_preprocessing.py .
COPY model_training.py .

# model_training.py imports df from data_preprocessing.py,
# so running it also runs the preprocessing step
CMD ["python", "model_training.py"]

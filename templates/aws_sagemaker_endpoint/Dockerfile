# Use an appropriate base image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


COPY . /opt/program
WORKDIR /opt/program

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x serve

# Expose the port for the FastAPI application
EXPOSE 8080

# Specify the command to run the inference script, no need to use ENTRYPOINT, this is for reference
# ENTRYPOINT ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080"]

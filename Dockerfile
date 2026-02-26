#use official Python image
FROM python:3.10-slim

#set working directory
WORKDIR /app

#copy requirement list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the project
COPY . .

#expose the API port
EXPOSE 8000

#command to run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
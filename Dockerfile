
FROM python:3.10-slim


WORKDIR /app


COPY requirements.txt requirements.txt


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000

# Command to run the application 
CMD ["python", "main.py"]

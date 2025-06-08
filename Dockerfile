FROM python:3.9-slim

LABEL maintainer="Kanishk <Kanishkgupta2003@outlook.com>"
LABEL version="1.0"
LABEL description="Renewable Data Helper"


# Set working directory
WORKDIR /RenewableDataHelper

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy project
COPY . .

# Expose port if your app uses one
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
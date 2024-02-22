from locust import HttpUser, task, between, SequentialTaskSet
import os

class PredictUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # Set the file path to the image you want to upload
        file_path = "sample/1.jpg"

        # Make sure the file exists
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        # Send a POST request with the image file
        with open(file_path, "rb") as file:
            self.client.post("/predict/", files={"file": ("1.jpg", file)})
# Run Locust with:
# locust -f locustfile.py

import requests

response = requests.post("https://ynoudjoukouang-wine1.herokuapp.com/predict_API", json={"input": [[6.6,0.16,0.4,1.5,0.044,48,143,0.9912,3.54,0.52,12.4]]})

print(response, response.json())



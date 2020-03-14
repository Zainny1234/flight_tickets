import requests

url = 'http://127.0.0.1:5000/predict_api'

data = {
    "Additional_Info": "No info",
    "Airline": "Jet Airways",
    "Arrival_Time": "04:25 07 Jun",
    "Date_of_Journey": "6/06/2019",
    "Dep_Time": "17:30",
    "Destination": "Cochin",
    "Duration": "10h 55m",
    "Route": "DEL → BOM → COK",
    "Source": "Delhi",
    "Total_Stops": "1 stop"
}
r = requests.post(url, json=data)
# r = requests.post("{}/".format("http://127.0.0.1:5000//predict_api"), json=data)
print(r.json())

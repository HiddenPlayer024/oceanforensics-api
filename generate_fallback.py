import requests
import json

def main():
    # Test case details
    region = {
        "min_lon": -10.0,
        "min_lat": 40.0,
        "max_lon": -9.0,
        "max_lat": 41.0
    }
    date_str = "2023-01-01"

    # 1. Call POST /detect
    detect_payload = {
        "region": region,
        "date": date_str
    }
    print("Sending request to /detect...")
    resp_detect = requests.post("http://localhost:8000/detect", json=detect_payload)
    resp_detect.raise_for_status()
    detect_data = resp_detect.json()
    print("Received /detect response.")

    # 2. Extract necessary fields for /hindcast
    centroid = detect_data.get("geometry", {}).get("centroid", {})
    polygon = detect_data.get("polygon", [])
    detection_timestamp = detect_data.get("timestamp", "")

    hindcast_payload = {
        "centroid": centroid,
        "polygon": polygon,
        "detection_timestamp": detection_timestamp,
        "region": region
    }

    # 3. Call POST /hindcast
    print("Sending request to /hindcast...")
    resp_hindcast = requests.post("http://localhost:8000/hindcast", json=hindcast_payload)
    resp_hindcast.raise_for_status()
    hindcast_data = resp_hindcast.json()
    print("Received /hindcast response.")

    # 4. Save combined output to demo_fallback.json
    output_data = {
        "test_parameters": {
            "region": region,
            "date": date_str
        },
        "detect_response": detect_data,
        "hindcast_response": hindcast_data
    }

    output_path = "/home/unknown/Projects/oil_spill_api/demo_fallback.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Successfully saved combined responses to {output_path}")

if __name__ == "__main__":
    main()

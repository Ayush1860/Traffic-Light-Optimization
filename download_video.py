import requests
import os

def download_video():
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/free-traffic-video-sample.mp4"
    output_path = "d:/Traffic-AI-Project/data/sample_traffic.mp4"
    
    if not os.path.exists("d:/Traffic-AI-Project/data"):
        os.makedirs("d:/Traffic-AI-Project/data")
        
    print(f"Downloading {url}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_video()

import requests

def test_transcribe_long_running_service():
    url = "http://0.0.0.0:5001/transcribe-long"
    files = {'file': open('/home/lab/Downloads/speech-trimmed_5min.wav', 'rb')}

    response = requests.post(url, files=files)

    assert response.status_code == 200
    assert 'results' in response.json()


    /home/lab/Downloads/speech-trimmed_5min.wav
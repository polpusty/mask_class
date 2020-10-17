import io
import zipfile
import requests


def download_model(url, save_path):
    """
    Download model files.
    """
    r = requests.get(url)
    bytesio_object = io.StringIO(r.text)
    with open(save_path, "w+") as f:
        f.write(bytesio_object.getbuffer())

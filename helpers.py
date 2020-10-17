import io
import zipfile
import requests


def download_model(url, save_path):
    """
    Download model files.
    """
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)

from urllib.request import urlretrieve
from os.path import isdir, join
from tqdm import tqdm
from zipfile import ZipFile

DATASET_URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
ZIPPED_FILENAME = 'dataset.zip'
DATA_DIR = 'EuroSAT_RGB'


def report(progress: tqdm):
    global progress_percent
    progress_percent = 0

    def reporter(count: int, block_size: int, total: int):
        global progress_percent
        current_percent = int(100*(count * block_size) / total)

        if (current_percent is not progress_percent):
            progress.update(current_percent - progress_percent)
            progress_percent = current_percent

    return reporter


def check_data() -> bool:
    return isdir(join('data', DATA_DIR))


def download_data():
    with tqdm(total=100) as progress_bar:
        urlretrieve(
            DATASET_URL,
            join('data', ZIPPED_FILENAME),
            report(progress_bar)
        )

    with ZipFile(join('data', ZIPPED_FILENAME)) as archive:
        archive.extractall('data')

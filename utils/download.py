from os.path import isfile, join
from urllib.request import urlretrieve
from os.path import isdir, join
from tqdm import tqdm
from zipfile import ZipFile


class DataLoader:
    def __init__(self, data_url, target_file, unzipped_target):
        self.loaded = isdir(join('data', unzipped_target))
        self.progress = 0
        self.data_url = data_url
        self.target_file = target_file

    def report(self, progress_bar: tqdm):
        def reporter(count: int, block_size: int, total: int):
            current_percent = int(100*(count * block_size) / total)

            if (current_percent is not self.progress):
                progress_bar.update(current_percent - self.progress)
                self.progress = current_percent

        return reporter

    def download_and_extract(self):
        with tqdm(total=100) as progress_bar:
            urlretrieve(
                self.data_url,
                join('data', self.target_file),
                self.report(progress_bar)
            )

        with ZipFile(join('data', self.target_file)) as archive:
            archive.extractall('data')

        self.loaded = True

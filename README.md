# A homemade convolutional neural network

This CNN can be used to clasify land cover from satellite images from the [EuroSAT dataset](https://github.com/phelber/EuroSAT).

## Run locally

#### Installation of the dependencies

- Conda has been used to manage environments
- The only dependencies are numpy and pillow (requires the [libjpeg prerequisite](https://pillow.readthedocs.io/en/stable/installation.html#external-libraries))

#### Getting the dataset

- Download the jpeg RGB dataset on [this hosting](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
- Unzip the dataset, move the containing folder to the project's root and rename it `raw_data`

#### Training the model

- Run `python main.py --should-preprocess` the first time to preprocess the data
- Just run `python main.py` the rest of the time

## Results

## Credits

This implementation was inspirated by :

- [TheIndependentCode](https://github.com/TheIndependentCode/Neural-Network)
- [Riccardo Andreoni](https://github.com/andreoniriccardo)

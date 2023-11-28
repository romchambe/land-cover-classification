# A homemade convolutional neural network

This CNN can be used to clasify land cover from satellite images from the [EuroSAT dataset](https://github.com/phelber/EuroSAT).

## Run locally

#### Installation of the dependencies

- Conda has been used to manage environments
- The only dependencies are numpy and pillow (requires the [libjpeg prerequisite](https://pillow.readthedocs.io/en/stable/installation.html#external-libraries))

#### Training the model

- Running `python main.py --should-preprocess` will download the dataset and preprocess the data
- Just run `python main.py` the rest of the time

## Results

The model achieves an accuracy of around 97% after 15 epochs

## Credits

This implementation was inspired by :

- [TheIndependentCode](https://github.com/TheIndependentCode/Neural-Network)
- [Riccardo Andreoni](https://github.com/andreoniriccardo)

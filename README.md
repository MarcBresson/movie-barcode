<div id="top"></div>


<br />
<div align="center">
  <img src="images/logo.png" alt="Logo" width="500" height="200">  

  <p align="center">
    <br />
    <a href="https://github.com/MarcBresson/movie-barcode/issues">Report Bug</a>
    ·
    <a href="https://github.com/MarcBresson/movie-barcode/issues">Request Feature</a>
  </p>

  <p align="center">
    <a href="https://github.com/MarcBresson/movie-barcode/issues"><img src="https://img.shields.io/github/issues/MarcBresson/movie-barcode.svg?style=for-the-badge"/></a>
    <a href="https://github.com/MarcBresson/movie-barcode/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/MarcBresson/movie-barcode.svg?style=for-the-badge"/></a>
    <a href="https://linkedin.com/in/marc--bresson"><img src="https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555"/></a>
  </p>
</div>



## About The Project

|            Fight Club - Intensity            |
|----------------------------------------------|
|![](./images/example-Fight_Club-intensity.png)|

|            Ponyo - Step            |
|------------------------------------|
|![](./images/example-Ponyo-step.png)|

|           SpiderMan: No way home - Intensity            |
|---------------------------------------------------------|
|![](./images/example-spiderman_no_way_home-intensity.png)|

|            Encanto - Hue            |
|-------------------------------------|
|![](./images/example-Encanto-hue.png)|

<p align="right">(<a href="#top">back to top</a>)</p>



## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, you can simply open an issue with the tag "enhancement". You can also fork the repo and create a pull request.
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



## Installation

Movie BarCode requires an installation of Python 3.6 or greater, as well as pip. (Pip is typically bundled with Python [installations](https://python.org/downloads).)

To install from PyPI with pip:
```bash
$ python -m pip install movie-barcode
```

Sometimes, the PyPI release becomes slightly outdated. To install from the source with pip:
```bash
$ python -m pip install git+https://github.com/MarcBresson/movie-barcode
```

<p align="right">(<a href="#top">back to top</a>)</p>



## Usage - QuickStart

Process a collection of movies at once :
```py
>>> from movie_barcode import Movies

# Default output directories are ./images and ./npy
>>> movies = Movies(images_output_dir, computed_colors_output_dir)

# Will open a dialog. If a movie has already been computed, it will recover the npy file.
>>> movies.load()
>>> movies.compute()
>>> movies.save() # optional. It allows to directly load the computed array on movies.load()
>>> movies.export_every_barcode()
```

<p align="right">(<a href="#top">back to top</a>)</p>



## License

Distributed under the EUPL 1.2 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



## Contact

Marc Bresson - marco.bresson@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>

from typing import List
from tkinter import filedialog, Tk
import logging
import os

from movie_barcode.movie import Movie

logger = logging.getLogger(__name__)

class Movies:
    def __init__(
        self,
        dir_output_images: str = './images',
        dir_output_np: str = './npy'
    ):
        self.dir_output_np = dir_output_np
        self.dir_output_images = dir_output_images
        self.movies: List[Movie] = []

    def load(
        self,
        paths: str | List[str] | None = None,
        check_for_existing = True
    ) -> None:

        if paths==None:
            root = Tk()
            root.withdraw()
            paths =  filedialog.askopenfilename(
                title = "Select files",
                filetypes = (
                    ("movie files",".mkv .mp4 .mpeg"),
                    ("all files","*.*")
                ),
                multiple=True
            )

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            self.movies.append(Movie(path))

        # We check if we already computed frames for every movie
        if check_for_existing:
            for movie in self.movies:
                movie.load(self.dir_output_np, "", True)

    def save(
        self
    ) -> None:
        
        os.makedirs(self.dir_output_np, exist_ok=True)

        for movie in self.movies:
            movie.save(
                self.dir_output_np,
            )

    def compute(
        self,
        id_video_flux: int = 0
    ) -> None:

        for movie in self.movies:
            movie.compute_colors(id_video_flux)
        
        if len(self.movies)==0:
            logger.error("No movie loaded. Colors generation aborted.")

    def export_barcode(
        self,
        method
    ) -> None:

        for movie in self.movies:
            movie.export_barcode(
                method,
                self.dir_output_images
            )

    def export_every_barcode(
        self
    ) -> None:

        for movie in self.movies:
            movie.export_every_barcode(
                self.dir_output_images
            )

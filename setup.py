from setuptools import find_packages, setup

setup(
    name="kitsu",
    version="0.0.3.4",
    description="",
    author="Kitsunetic",
    author_email="jh.shim.gg@gmail.com",
    url="https://github.com/Kitsunetic/kitsu.git",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "scikit-image",
        # "point_cloud_utils",
        # "PyMCubes",
        "omegaconf",
        "easydict",
        "tqdm",
        "matplotlib",
    ],
    entry_points={"console_scripts": ["kitsu=kitsu.main:main"]},
)

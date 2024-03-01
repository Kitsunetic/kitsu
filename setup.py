from setuptools import find_packages, setup

setup(
    name="kitsu",
    version="0.1.1",
    description="",
    author="Kitsunetic",
    author_email="jh.shim.gg@gmail.com",
    url="https://github.com/Kitsunetic/kitsu",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        # "torch",
        # "numpy",
        # "point_cloud_utils",
        "scikit-image",
        "PyMCubes",
        "omegaconf",
        "easydict",
        "tqdm",
        "matplotlib",
    ],
    # entry_points={"console_scripts": ["kitsu=kitsu.main:main"]},
)

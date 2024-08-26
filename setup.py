from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    desc = f.read()

setup(
    name="kitsu",
    version="0.1.2.5",
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
        # "PyMCubes",
        "omegaconf",
        "easydict",
        "tqdm",
        "matplotlib",
    ],
    # entry_points={"console_scripts": ["kitsu=kitsu.main:main"]},
    long_description=desc,
    long_description_content_type="text/markdown",
)

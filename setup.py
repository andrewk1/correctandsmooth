from setuptools import setup, find_packages

setup(name="cands",
      version="0.0.1",
      packages=find_packages(),
      install_requires=[
        "torch>=1.6.0",
        "tqdm>=4.27",
        "numpy"
      ],
      )

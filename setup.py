from setuptools import setup, find_packages


setup(
   name='kaggle_SIIM_ISIC',
   version='1.0',
   description='Personal project package',
   author='atgm1113',
   author_email='atgm1113@gmail.com',
   packages=find_packages(include=["src", "src.data_process", "src.main", "src.model"], exclude=["data"]))
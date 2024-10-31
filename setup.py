from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='avg',
      packages=find_packages([package for package in find_packages()
                if package.startswith('avg') or package.startswith('test')]),
      install_requires=[],
      description='Action Value Gradient algorithms',
      author='Gautham Vasan',
      url=' https://github.com/gauthamvasan/avg',
      author_email='gauthamv.529@gmail.com',
      version='0.0.1')

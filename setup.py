from setuptools import setup

setup(name='ibv',
      version='0.1',
      description='Innate Binocular Vision Tools',
      url='http://github.com/binocular-vision/ibv',
      author='Samuel Sendelbach',
      author_email='ssendelbach@luc.edu',
      license='MIT',
      packages=['ibv'],
      install_requires=[
          'Pillow',
          'google-cloud-storage',
          'numpy',
          'scipy',
          'sklearn',
      ],
      zip_safe=False)

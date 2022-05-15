from setuptools import setup, find_packages

setup(
  name = 'autocards',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'autocard = autocards.cli:main'
    ],
  },
  version = '0.0.1',
  license='MIT',
  description = 'Accelerating learning through machine-generated flashcards.',
  author = 'Paul Bricman',
  author_email = 'paulbricman@protonmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/ncoop57/Autocards/tree/master',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'text to image'
  ],
  install_requires=[
    'PyPDF2 == 1.26.0',
    'beautifulsoup4 == 4.9.3',
    'fastcore == 1.4.2',
    'nltk == 3.5',
    'pandas == 1.2.3',
    'protobuf == 3.20.1',
    'requests == 2.24.0',
    'sentencepiece == 0.1.96',
    'tika == 1.24',
    'torch == 1.8.1',
    'tqdm == 4.55.1',
    'transformers == 4.19.1',
    'epub-conversion == 1.0.15',
    'xml_cleaner == 2.0.4'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
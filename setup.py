from skbuild import setup
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="simlar",
    version="0.1",
    include_package_data=True,
    author=['Junjie Xia, Kazu Terao'],
    description='Simulation for LArTPC',
    license='MIT',
    keywords='simlar',
    scripts=['bin/simlar-config.py','bin/simlar-run.py','bin/simlar-filemerger.py'],
    packages=['simlar'],
    package_data={'simlar': ['config/*.yaml']},
    install_requires=[
        'h5py',
        'pyyaml',
        'numpy',
        'scikit-build',
        'torch',
        'gdown',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)

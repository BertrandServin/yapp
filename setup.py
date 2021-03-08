from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup( name = 'yapp',
       version = '0.1',
       description = 'Yet Another Phasing Program',
       long_description = readme(),
       url = 'https://github.com/BertrandServin/yapp',
       author = 'Bertrand Servin',
       author_email = 'bertrand.servin@inrae.fr',
       license = 'LGPLv2.1',
       packages = ['yapp'],
       install_requires = [
           'numpy',
           'scipy',
           'cyvcf2',
           'zarr',
           'h5py',
           'fastphase'
           ],
       scripts=['bin/fphtrain','bin/yapp'],
       zip_safe = False)
       

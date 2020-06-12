from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup( name = 'yapp',
       version = '0.1',
       description = 'Yet Another Phasing Project',
       long_description = readme(),
       url = 'https://github.com/BertrandServin/yapp',
       author = 'Bertrand Servin',
       author_email = 'bertrand.servin@inrae.fr',
       license = 'LGPLv3',
       packages = ['yapp'],
       install_requires = [
           'numpy',
           'cyvcf2',
           'fastphase>=2'
           ],
       scripts=['bin/fphtrain'],
       zip_safe = False)
       

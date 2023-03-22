from setuptools import setup, find_packages
setup(name='kid_model',
      version='0.0',
      description='Mattis-Bardeen model of KID response and noise',
      author='Elijah Kane',
      author_email='ekane@caltech.edu',
      packages=find_packages(),
      url="https://github.com/ekane123/kid_model",
      data_files = [],
      include_package_data=True,
      python_requires='>3.8',
      install_requires=['numpy<1.23,>=1.18',
                        'scipy',
                        'pandas',
                        'matplotlib',
                        #'pyfftw'
                        #'submm_python_routines @ git+ssh://git@github.com/Wheeler1711/submm_python_routines'
                        ]
      )

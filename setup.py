from setuptools import setup

setup(
    name='instgraph2vec',
    version='0.1.0',    
    description='A interface for Graph2Vec',
    url='https://github.com/maximkha/instgraph2vec',
    author='Maxim Khanov',
    author_email='maximkha@outlook.com',
    license='GPLV3',
    packages=['instgraph2vec'],
    install_requires=['gensim',
                      'tqdm',
                      'networkx',
                      'pandas',
                      'joblib'
                      ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
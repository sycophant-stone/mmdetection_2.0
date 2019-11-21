from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='mmdet-pipeline',
        version="v1.0",
        description='mmdetection evalution pipeline',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='Apache License 2.0',
        install_requires=[
            'prettytable'
        ],
        zip_safe=False)
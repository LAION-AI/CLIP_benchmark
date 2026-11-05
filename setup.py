#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

def load_requirements(f):
    return [l.strip() for l in open(f).readlines() ]

requirements = load_requirements("requirements.txt")

test_requirements = requirements + ["pytest", "pytest-runner"]

setup(
    author="Mehdi Cherti",
    author_email='mehdicherti@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="CLIP-like models benchmarks on various datasets",
    entry_points={
        'console_scripts': [
            'clip_benchmark=clip_benchmark.cli:main',
            'clip_benchmark_export_wds=clip_benchmark.webdataset_builder:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='clip_benchmark',
    name='clip_benchmark',
    packages=find_packages(include=['clip_benchmark', 'clip_benchmark.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mehdidc/clip_benchmark',
    version='1.6.2',
    zip_safe=False,
    extras_require = {
        "vtab": ["task_adaptation==0.1", "timm>=0.5.4"],
        "tfds": ["tfds-nightly", "timm>=0.5.4"],
        "coco": ["pycocotools>=2.0.4"],
        # audio models are only on OpenCLIP's main branch, MSEB is not released at this version, and the k-means of
        # MSEB's clustering gives other results with other scikit-learn versions
        "audio": ["open_clip_torch[audio] @ git+https://github.com/mlfoundations/open_clip.git@7067b350050dc409bfdfdbb3f0cc6c9a1651c000",
                  "datasets>=5.0.1,<6", "mteb[audio]==2.21.0", "scikit-learn==1.9.1",
                  "mseb @ git+https://github.com/google-research/mseb.git@42595cf4fb55a96adfc77ace544fb19d27490a3b"],
        # the dependencies of heareval, which pins outdated ones itself: install it with --no-deps, see README.
        # Lightning 1.6-1.8 keeps heareval's best-epoch selection, dcase_util needs pkg_resources (setuptools<82),
        # and newer pynvml returns str where heareval expects bytes
        "hear": ["tensorflow", "intervaltree", "pytorch-lightning>=1.6,<1.9", "setuptools<82", "pynvml<11.5", "dcase_util",
                 "sed_eval", "python-slugify", "torchinfo", "more-itertools", "click"],
    }
)

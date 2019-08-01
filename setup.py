from distutils.core import setup

setup(
    name='dump1090mapper',
    version='0.1.0',
    author='Jeremy B. Merrill',
    author_email='jeremybmerrill@jeremybmerrill.com',
    packages=['dump1090mapper', 'dump1090mapper.test'],
    scripts=['mapify.py'],
    # url='http://pypi.python.org/pypi/TowelStuff/',
    license='MIT',
    description='make maps of planes (in NYC)',
    long_description=open('README.txt').read(),
    install_requires=[],
)

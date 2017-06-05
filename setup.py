from distutils.core import setup

setup(name='multiplexcd',
		version='1.0',
		description='Multiplex Community Detection',
		author='Michael Siemon',
		author_email='mcs296@cornell.edu',
		url='https://github.com/michaelsiemon/multiplexcd',
		py_modules=['multiplexcd'],
		license='MIT',
		keywords='networks graphs community detection multiplex',
		install_requires=['igraph', 'numpy', 'scipy']
		)
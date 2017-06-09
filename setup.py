from distutils.core import setup

setup(name='multiplexcd',
		version='1.0',
		description='Functions to peform multiplex community detection',
		author='Michael Siemon',
		author_email='mcs296@cornell.edu',
		url='https://github.com/michaelsiemon/multiplexcd',
		py_modules=['multiplexcd'],
		license='MIT',
		keywords='network graph community detection multiplex',
		requires=['igraph', 'numpy', 'scipy']
		)
import sys 
try:
	from setuptools import setup
	have_setuptools = True 
except ImportError:
	from distutils.core import setup 
	have_setuptools = False

setup_kwargs = {
'name': 'HpyRoutines', 
'version': '0.1.0', 
'description': 'Few hopefully helpful Healpy-based routines', 
'author': 'Federico Bianchini', 
'author_email': 'federico.bianxini@gmail.com', 
'url': 'https://github.com/fbianchini/HpyRoutines', 
'packages':['hpyroutines'],
'zip_safe': False, 
}

if __name__ == '__main__': 
	setup(**setup_kwargs)
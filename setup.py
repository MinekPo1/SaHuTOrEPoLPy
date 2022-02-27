import pathlib
from setuptools import setup
from sahutorepol.Types import check_compatability

check_compatability()

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Check if simple_warnings is bundled

bundle_simple_warnings = (HERE / "simple_warnings").exists()

# This call to setup() does all the work
setup(
	name="SaHuTOrEPoLPy",
	version="0.0.5.3",
	description="SaHuTOrEPoLPy is a simple interpreter for SaHuTOrEPoL",
	long_description=README,
	long_description_content_type="text/markdown",
	url="https://github.com/MinekPo1/SaHuTOrEPoLPy",
	author="MinekPo1",
	author_email="minekpo1@gmail.com",
	license="fafol",
	classifiers=[
		"License :: Free To Use But Restricted",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.10",
	],
	packages=["sahutorepol"]
		+ (["simple_warnings"] if bundle_simple_warnings else []),
	include_package_data=True,
	install_requires=["typing","click"]
		+ ([] if bundle_simple_warnings else ["simple_warnings"]),
	entry_points={
		"console_scripts": [
			"sahutorepol=sahutorepol.__main__:cli",
		]
	},
)

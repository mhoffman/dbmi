#!/usr/bin/env python

from distutils.core import setup
import os
from dbmi import __version__ as version

maintainer = 'Max J. Hoffmann'
maintainer_email = 'mjhoffmann@gmail.com'
author = 'Max J. Hoffmann'
author_email = 'mjhoffmann@gmail.com'
name = 'dbmi'
description =  __doc__

classifiers = []
requires = [
                    'ase',
                    'espresso',
                   ]

package_dir = {'dbmi': 'dbmi'}
package_data = {}
packages = ['dbmi']
platforms = ['linux', 'windows', 'darwin']
url = 'https://github.com/mhoffman/dbmi'
long_description = file('README.md').read()
scripts = []

setup(author=author,
      author_email=author_email,
      description=description,
      requires=requires,
      license=license,
      long_description=long_description,
      maintainer=maintainer,
      maintainer_email=maintainer_email,
      name=name,
      package_data=package_data,
      package_dir=package_dir,
      packages=packages,
      platforms=platforms,
      scripts=scripts,
      url=url,
      version=version,
      )

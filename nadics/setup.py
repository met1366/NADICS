from setuptools import find_packages, setup
import version as ver

REQUIRED_PACKAGES = [
        'pandas>=0.21.1',
        'scikit-learn>=0.19.1',
        'scipy>=1.0.0',
        'sklearn>=0.0',
        'matplotlib>=2.1.1',
        'imbalanced-learn>=0.3.2',
        'imblearn>=0.0'
]

CONSOLE_SCRIPTS = [
        'nadics = nadics.main:run_main',
]

setup(name="NADICS",
      version=ver.VERSION,
      description="Machine Learning Engine for NIDS",
      author="Mohammad Norouzian, Fabian Weise",
      author_email="norouzian@sec.in.tum.de, fabian.weise@tum.de",
      packages=find_packages(),
      entry_points={
          'console_scripts': CONSOLE_SCRIPTS,
      },
      python_requires='>= 2.7, != 3.0.*, != 3.1.*',
      install_requires=REQUIRED_PACKAGES
      )

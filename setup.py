#  Copyright 2024 Kohei Suzuki
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys

try:
    from skbuild import setup
except ImportError:
    from setuptools import setup

setup_requires = [
    "numpy < 1.27.0",
    "psutil",
    "h5py",
    "pybind11 >=2.11.0, < 2.12.0",
    "cmake > 3.20",
    "scikit-build > 0.16.0",
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

setup(
    setup_requires=setup_requires,
    use_scm_version=True,
    packages=[  
        'compnal',
        'compnal.lattice',
        'compnal.model',
        'compnal.model.classical',
        'compnal.solver',
        'compnal.utility',
        ],
    cmake_install_dir="compnal",
    include_package_data=False,
    zip_safe=False,
)

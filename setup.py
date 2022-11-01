# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from datetime import datetime
from setuptools import setup, PEP420PackageFinder


def get_version():
    version = os.environ['TRANSFORMERS_NEURONX_VERSION']
    today = datetime.today().strftime('%Y%m%d')
    return version.replace('.x', f'.{today}')


setup(
    name='transformers-neuronx',
    version=get_version(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='aws neuron neuronx transformers',
    packages=PEP420PackageFinder.find(where='src'),
    package_data={
        'transformers_neuronx': [
            'LICENSE',
        ],
    },
    entry_points = {
        'console_scripts': [
            'gpt2_demo=transformers_neuronx.gpt2.demo:main',
            'gptj_demo=transformers_neuronx.gptj.demo:main',
            'opt_demo=transformers_neuronx.opt.demo:main',
            'gen_random_opt_175b=transformers_neuronx.opt.gen_random_175b:main',
            'gen_randn_hlo_snapshot=transformers_neuronx.tools.gen_hlo_snapshot:main_randn',
        ],
    },
    install_requires=[
        'accelerate',
        'torch-neuron',  # TODO: point to torch-neuronx
        'transformers',
    ],
    python_requires='>=3.7',
    package_dir={'': 'src'},
)

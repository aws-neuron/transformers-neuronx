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

def maybe_override_attributes(self, kwargs):
    for key, value in kwargs.items():
        if not hasattr(self, key):
            raise KeyError(f'Found invalid key "{key}".')
        if value is not None:
            setattr(self, key, value)


def power_of_two_bucket_sizes(min_bucket_size, max_bucket_size):
    sizes = []
    bucket_size = min_bucket_size
    while bucket_size < max_bucket_size:
        sizes.append(bucket_size)
        bucket_size *= 2
    sizes.append(max_bucket_size)
    return sizes


def pad_vocab_size(vocab_size, tp_degree):
    return ((vocab_size // tp_degree + 1) * tp_degree - vocab_size) % tp_degree

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
import bisect
from typing import List, Union, Optional

from transformers_neuronx import utils


def token_sizes(buckets_or_size: Union[List[int], int]) -> List[int]:
    """
    Compute the bucket sizes for token generation.

    The `buckets_or_size` argument may be specified as a list of buckets and
    any other logic in this function will be ignored.

    When `buckets_or_size` is an integer value, buckets will be chosen by
    doubling the size of each bucket starting at 128 up to the provided value.

    Arguments:
        buckets_or_size: A list of buckets or maximum size for token generation.

    Returns
        buckets: The list of bucket sizes for token generation.
    """
    if isinstance(buckets_or_size, list):
        return sorted(buckets_or_size)
    return utils.power_of_two_bucket_sizes(128, buckets_or_size)


def context_sizes(
        buckets_or_size: Optional[Union[List[int], int]],
        token_buckets: Optional[List[int]] = None,
    ) -> List[int]:
    """
    Compute the bucket sizes for context encoding.

    The `buckets_or_size` argument may be specified as a list of buckets or
    a single bucket and any other logic in this function will be ignored.

    When `bucket_or_size` is None, the default context length bucket sizes
    are set to be half of token model bucket sizes.

    When `bucket_or_size` is set to 0 this completely disables context
    encoding.

    Arguments:
        buckets_or_size: A list of buckets or maximum size for context encoding.
        token_buckets: The token buckets to generate context buckets for.

    Returns
        buckets: The list of bucket sizes for context encoding.
    """
    if isinstance(buckets_or_size, list):
        return sorted(buckets_or_size)
    if isinstance(buckets_or_size, int):
        if buckets_or_size <= 0:
            return []
        return [buckets_or_size]
    if token_buckets is not None and buckets_or_size is None:
        return [bucket // 2 for bucket in token_buckets]
    raise NotImplementedError(f'Prompt bucket config {buckets_or_size} not supported')


def batch_sizes(batch_size: Union[List[int], int]) -> List[int]:
    """
    Format the user-specified batch size buckets for all model variants.

    Arguments:
        batch_size: The batch size(s) to construct models for.

    Returns
        batch_sizes: The formatted list of batch sizes.
    """
    if isinstance(batch_size, int):
        return [batch_size]
    elif isinstance(batch_size, list):
        return sorted(batch_size)
    else:
        raise TypeError("batch_size must be list of ints or int type")


def find(buckets: Optional[List[int]], size: int) -> Optional[int]:
    """
    Find the smallest bucket with that fits the given `size` input.

    When `size` exceeds the largest bucket size, the largest bucket will be
    returned.

    Arguments:
        buckets: Either the prompt/token bucket sizes to search.
        size: The size to fit into the buckets.

    Return:
        bucket: The bucket value (not the bucket index)
    """
    if not buckets:  # When we have no buckets, return None
        return None
    index = bisect.bisect_left(buckets, size, hi=len(buckets) - 1)
    return buckets[index]

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
import enum

# Tile size used for the weight transformation
TILE_SIZE = 128

# Size used to determine fused QKV operation.
FUSED_QKV_TP_FACTOR = 3

# KV sharding pad for flash decoding
KV_SHARD_PAD = 128

# Number of chips on Trn
TRN1_WORLD_SIZE = 32

# Layout for attention
LAYOUT_BSH = 'BSH'
LAYOUT_HSB = 'HSB'
LAYOUT_SBH = 'SBH'


# fp8 bounds
class FpBounds:
    min = -240.0
    max = 240.0


class Layout(enum.Enum):
    HSB = 'HSB'
    BSH = 'BSH'
    SBH = 'SBH'

    def __eq__(self, value):
        return super().__eq__(Layout(value))


# Group query attention sharding configurations
class GQA(enum.Enum):
    # [Default] Sharding over the heads splits entire (complete) K/V heads
    # onto the NeuronCores where the corresponding Q heads reside. This is
    # similar to traditional MHA except that the Q and K/V heads do not need
    # to be equal.
    #
    # This cannot be enabled when number of K/V heads cannot be evenly split
    # across the NeuronCores according to the tensor parallelism degree.
    SHARD_OVER_HEADS = 'shard-over-heads'

    # Sharding over the bach dimension linearly shards the K/V heads across
    # all NeuronCores (incomplete heads per NeuronCore) and shards the K/V
    # cache across the batch dimension according to the tensor parallellism.
    # These partial K/V heads are concatenated and then split across NeuronCores
    # along the batch dimension (AllToAll) before computing the attention.
    #
    # This cannot be enabled when the batch size cannot to be evenly split
    # across the NeuronCores according to the tensor parallelism degree.
    SHARD_OVER_BATCH = 'shard-over-batch'

    # This transforms a GQA attention mechanism into a traditional MHA mechanism
    # by replicating the K/V heads to evenly match the corresponding Q heads.
    # This consumes more memory than would otherwise be used with other sharding
    # mechanisms but avoids collective communications overheads.
    REPLICATED_HEADS = 'replicated-heads'

    # This mechanism evenly splits the K/V heads across all NeuronCores
    # (incomplete heads per NeuronCore). These partial k/V heads are
    # concatenated to the the NeuronCores where the corresponding Q heads
    # reside (AllGather). This can be more memory efficient than replication
    # but introduces an additional collective communication operation per
    # decoder layer.
    #
    # This cannot be enabled when the number of Q heads cannot to be evenly
    # split across the NeuronCores according to the tensor parallelism degree.
    ALL_GATHER_HEADS = 'all-gather-heads'

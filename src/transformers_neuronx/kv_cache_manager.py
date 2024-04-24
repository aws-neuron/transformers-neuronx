
import torch

from transformers_neuronx import parallel
from transformers_neuronx import compiler
from transformers_neuronx import decoder
from transformers_neuronx import ops
from transformers_neuronx import hlo

class KVCacheManager:

    def __init__(self, cache_shape, tp, num_layer, dtype=torch.float32):
        self.cache_shape = cache_shape

        self.tp = tp
        self.num_layer = num_layer
        self.manipulator = parallel.ParallelTensorManipulator(self.tp)

        self.dtype = dtype


    def generate_cache(self, generator=torch.zeros):
        cache = generator(self.cache_shape).to(self.dtype)

        ops.init()
        k_caches = [(self.manipulator.shard_along(cache, dim=2)) for _ in range(self.num_layer)]
        v_caches = [(self.manipulator.shard_along(cache, dim=2)) for _ in range(self.num_layer)]

        self.set_kv_cache(k_caches, v_caches)


    def set_kv_cache(self, k_caches, v_caches):
        assert len(k_caches) == len(v_caches)

        self.k_caches = k_caches
        self.v_caches = v_caches

    def set_cache_shape(self, cache_shape):
        self.cache_shape = cache_shape
        self.n_positions, self.batch_size, self.n_heads_kv_cache, self.attention_head_size = cache_shape


    def to_cpu(self):
        cpu_k_caches = [(self.manipulator.unshard_along(cache, dim=2)) for cache in self.k_caches]
        cpu_v_caches = [(self.manipulator.unshard_along(cache, dim=2)) for cache in self.v_caches]

        return cpu_k_caches, cpu_v_caches

    def send_kv_cache_program(self, source_g_start_id, target_g_start_id, world_size):

        def send_program(scribe):
            param_builder = decoder.DecoderParameterBuilder(scribe, 0)
            cache_params = []
            for cache in self.k_caches:
                cache_param = param_builder.from_tensor(cache)
                cache_params.append(cache_param)

            for cache in self.v_caches:
                cache_param = param_builder.from_tensor(cache)
                cache_params.append(cache_param)

            # concat on batch dim, easier to slice
            cache_shape = cache_params[0].sizes
            concat_size = [cache_shape[0], len(cache_params) * cache_shape[1], *cache_shape[2:]]
            cat_cache = cache_params[0].dtype[concat_size].Concatenate(*cache_params, dimensions=[1])

            return hlo.all_reduce_sum(
                cat_cache,
                tp_degree=self.tp,
                dtype=cat_cache.dtype,
                replica_groups=[[source_g_start_id+i, target_g_start_id+i] for i in range(self.tp)]
            )

        self.send_hlo_kernel = compiler.HLOKernel(send_program, self.tp, start_g_nc_id=source_g_start_id, g_nc_count=world_size*self.tp)
        self.send_hlo_kernel.build()
        self.send_hlo_kernel.load()
        self.send_hlo_kernel.setup([*self.k_caches, *self.v_caches], [])

    def run_send(self):
        self.send_hlo_kernel.run()
        return self.manipulator.unshard_along(self.send_hlo_kernel.memories.output_tensors[0], dim=2)


    def receive_kv_cache_program(self, source_g_start_ids, g_start_id, batch_ids, source_batch_size=1):

        world_size = len(source_g_start_ids) + 1
        def receive_program(scribe):
            param_builder = decoder.DecoderParameterBuilder(scribe, 0)
            cache_params = []
            for cache in self.k_caches:
                cache_param = param_builder.from_tensor(cache)
                cache_params.append(cache_param)

            for cache in self.v_caches:
                cache_param = param_builder.from_tensor(cache)
                cache_params.append(cache_param)


            # concat on batch dim, easier to slice
            cache_shape = cache_params[0].sizes
            cache_dtype = cache_params[0].dtype
            concat_size = [cache_shape[0], len(cache_params) * source_batch_size, *cache_shape[2:]]

            for source_g_start_id, batch_id in zip(source_g_start_ids, batch_ids):

                zero = cache_dtype.Constant(constant_value=0)
                cat_cache = cache_dtype[concat_size].Broadcast(zero, dimensions=[])

                cat_cache = hlo.all_reduce_sum(
                    cat_cache,
                    tp_degree=self.tp,
                    dtype=cat_cache.dtype,
                    replica_groups=[[source_g_start_id+i, g_start_id+i] for i in range(self.tp)]
                )

                for i in range(2): # k,v
                    for j in range(self.num_layer):
                        offset_in_cat = i*self.num_layer + j # offset in cat kv cache
                        cache_slice = hlo.slice_along(cat_cache, 1, offset_in_cat*source_batch_size + source_batch_size, offset_in_cat*source_batch_size)
                        start_indices = [0, batch_id, 0, 0]

                        updated_cache = hlo.dynamic_update_slice(cache_params[offset_in_cat], cache_slice, start_indices)
                        cache_params[offset_in_cat] = updated_cache

            root_shapes = [cache.dtype[cache.sizes] for cache in cache_params]
            return scribe.tuple(*root_shapes).Tuple(*cache_params)

        self.recv_hlo_kernel = compiler.HLOKernel(receive_program, self.tp, start_g_nc_id=g_start_id, g_nc_count=self.tp*world_size)
        self.recv_hlo_kernel.build()
        self.recv_hlo_kernel.load()

        # update cache inplace
        # we cannot simply slice cache[line_id] and update inplace
        # as the slice dimension is not the first one, leads to non-contiguous slice
        cache_buffers = [*self.k_caches,*self.v_caches]

        self.recv_hlo_kernel.setup(cache_buffers, cache_buffers)

    def run_receive(self):
        self.recv_hlo_kernel.run()

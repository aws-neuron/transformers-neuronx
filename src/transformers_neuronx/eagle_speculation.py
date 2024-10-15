import itertools
from typing import Optional, List, Dict

import torch

from transformers_neuronx import compiler
from transformers_neuronx import utils
from transformers_neuronx import hlo
from transformers_neuronx import decoder
from transformers_neuronx import program
from transformers_neuronx import base
from transformers_neuronx.layers.attention import fused_kv_update_cache
from transformers_neuronx.util.token_tree import validate_token_tree, generate_attention_mask
from transformers_neuronx.config import GenerationConfig
from transformers_neuronx.fused_speculation import FusedSpeculativeBase, PseudoAcceptorSampler


class EagleSpeculativeDecoder(FusedSpeculativeBase):
    """
    A speculative decoder which fuses the compute of the draft & target models.

    This is based on the original DeepMind paper.
    Reference: https://arxiv.org/pdf/2302.01318.pdf

    Compared with the regular `SpeculativeGenerator`, the purpose of this
    implementation is to avoid the typical overheads associated with CPU
    sampling. This can be especially impactful when using very fast draft
    models.

    Unlike the CPU sampling implementation, this version of the speculative
    decoder *always* executes k + 1 draft iterations in order to populate
    the draft model KV cache invariant to the number of token rejections. This
    means that this implementation may perform worse compared to sampling
    performed on CPU if the `draft` model is relatively slow and if the `k` value
    is small. This is because the CPU implementation has the ability to skip
    the final draft execution when there is at least 1 rejection.

    Arguments:
        draft: A fast and less accurate model to perform `k` speculations with.
        target: A slower and more accurate model which consumes speculated
            tokens.
        k: The number of tokens to speculate with the `draft` model.
        eos_token_id: The identifier for the end of sentence token. When
            provided, early stopping will be enabled. Otherwise the sample loop
            will continue until the maximum sequence length is reached.
        pad_token_id: The identifier which is used to pad batches of uneven
            sequence lengths. This token should be excluded from the
            generation results. This should be explicitly specified when
            using a `streamer`.
        buckets: An optional number of buckets to compile the fused speculative
            model for. By default, the number of buckets used will be derived
            from the `target` model.
        output_scores: Flag that indicates whether to construct the fused model
            so that it will return the target model scores during sampling.
    """

    def __init__(
            self,
            draft: base.NeuronModelBase,
            target: base.NeuronModelBase,
            k: int = 2,
            pad_token_id: int = 0,
            eos_token_id: Optional[int] = None,
            buckets: Optional[List[int]] = None,
            output_scores: Optional[bool] = False,
            token_tree: Optional[Dict[int, List[int]]] = None,
            greedy: Optional[bool] = False,
            debug: Optional[bool] = False,
            simulation_acceptance_length: Optional[float] = None,
        ) -> None:
        super().__init__()

        assert draft.neuron_config.on_device_embedding == True, (
            "The draft model must enable on-device embedding."
        )
        assert draft.neuron_config.on_device_generation is not None, (
            "The draft model must enable on-device sampling."
        )
        assert draft.neuron_config.use_2d_cache_ids, (
            "We only support 2D cache ID. Set padding_side='right'."
        )
        assert target.neuron_config.on_device_embedding == True, (
            "The target model must enable on-device embedding."
        )
        assert target.neuron_config.on_device_generation is not None, (
            "The target model must enable on-device sampling."
        )
        assert target.config.vocab_size == draft.config.vocab_size, (
            "The target model and draft model must have the same vocab size."
        )
        assert target.neuron_config.use_2d_cache_ids, (
            "We only support 2D cache ID. Set padding_side='right'."
        )
        # TODO: See if there is a way we can enable different tp degrees
        assert target.decoder_lm_head.tp_degree == draft.decoder_lm_head.tp_degree, (
            "The target model and draft model must have the same tp degree."
        )
        assert token_tree is not None, (
            "Need to provide a token tree."
        )

        # FIXME: Add more validation to ensure draft/target compatibility

        # User-provided attributes
        self.draft = draft
        self.target = target
        self.token_tree = token_tree
        if token_tree is not None:
            self.k, self.depth = validate_token_tree(token_tree)
            self.width = self._get_token_tree_width()
            self.tree_matrix = self._get_tree_matrix()
            self.token_paths, *rst = self._get_all_paths()
            self.n_leaves = self.token_paths.shape[0]
        else:
            self.k = k
            self.depth = k
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_scores = output_scores
        self.greedy = greedy
        self.debug = debug

        self.sampler = None
        if simulation_acceptance_length is not None:
            self.sampler = PseudoAcceptorSampler(simulation_acceptance_length)

        # Derived attributes
        self.neuron_config = self.target.neuron_config
        self.tp_degree = self.target.decoder_lm_head.tp_degree
        if buckets is None:
            buckets = self.target.decoder_lm_head.n_positions_list
        self.buckets = buckets
        self.batch_sizes = self.target.batch_sizes
        self.max_position = buckets[-1]
        self.vocab_size = self.target.config.vocab_size

        # Internal attributes
        self.speculator = None
        self.hidden = {}
        self.cache_gather_indices = {}
        self.cache_scatter_indices = {}

    def _get_token_tree_width(self):
        """
        Helper function to determine the topk value for each draft execution
        """
        assert self.token_tree is not None

        width = 1
        for v in self.token_tree.values():
            width = max(width, len(v))
        return width

    def _get_update_indices(self, batch_size):
        """
        Helper function for getting the gather indices during draft execution
        Each draft execution results in topk results, and we only need some of the topk results
        """
        assert self.token_tree is not None

        update_indices = torch.zeros(self.k-1, dtype=torch.int64)
        for k, v in self.token_tree.items():
            for i in v:
                offset = i - v[0]
                update_indices[i-1] = k * self.width + offset
        return update_indices.expand(batch_size, -1)

    def _get_hidden_update_indices(self, batch_size):
        """
        Helper function for getting the gather indices during draft execution
        Similar helper function as _get_update_indices() but for hidden states
        """
        assert self.token_tree is not None

        hidden_update_indices = torch.zeros(self.k-1, dtype=torch.int64)
        for k, v in self.token_tree.items():
            for i in v:
                hidden_update_indices[i-1] = k
        return hidden_update_indices.expand(batch_size, -1)

    def _get_position_ids(self, batch_size, cache_ids):
        """
        Helper functions for constructing position_ids, which is important for Llama rotary embedding
        The nodes at the same level of the tree should have the same position_id
        """
        offsets = cache_ids.clone()[:, :1]
        ids = torch.zeros((batch_size, self.k), dtype=torch.int64)

        def dfs(k, depth):
            nonlocal ids

            if k not in self.token_tree.keys():
                return
            for v in self.token_tree[k]:
                ids[:, v] = depth + 1
                dfs(v, depth+1)

        dfs(0, 0)

        ids += offsets

        return ids

    def _get_tree_matrix(self):
        """
        Helper function for getting the matrix representing the flattened tree structure
        Each entrance/row corresponds to a node in the tree
        Each element of the row represents the child of the node

        Example:
        tree: {0: [1, 2], 1: [3, 4], 2: [5], 3: [6]}
        tree_matrix
        [[1, 2],
         [3, 4],
         [5, 0],
         [6, 0],
         [0, 0],
         [0, 0],
         [0, 0]]
        """
        assert self.token_tree is not None

        matrix = torch.zeros((self.k, self.width), dtype=torch.int64)

        for k in self.token_tree.keys():
            for i in range(len(self.token_tree[k])):
                matrix[k, i] = self.token_tree[k][i]

        return matrix

    def _get_node_indices(self):
        """
        Helper function for reconstructing the probabilities from tree matrix
        Each prob in the element represents the prob of a node in the tree

        Example:
        tree: {0: [1, 2], 1: [3, 4], 2:[5], 3: [6]}
        flattened_probs:
        [[1, 2, 3, 4, 5, 0, 6, 0, ...]]
        node_indices: [1, 2, 3, 4, 5, 7]
        """
        indices = torch.zeros((self.k-1,), dtype=torch.int64)

        for k in self.token_tree.keys():
            offset = self.token_tree[k][0]
            for v in self.token_tree[k]:
                indices[v-1] = self.width*k + v - offset

        return indices

    def _get_all_paths(self):
        """
        Helper function for getting all possible paths from a tree
        A mask is also returned for specifying which values are valid

        Example:
        tree: {0: [1, 2], 1: [3, 4], 2:[5], 3: [6]}
        all_path:
        [[0, 1, 3, 6],
         [0, 1, 4, 0],
         [0, 2, 5, 0]]
        mask:
        [[True, True, True, True],
         [True, True, True, False],
         [Tuee, True, True, False]]
        """
        paths = []
        max_len = 1

        def dfs(k, path):
            nonlocal paths
            nonlocal max_len

            if k not in self.token_tree.keys():
                paths.append(path.copy())
                if len(path) > max_len:
                    max_len = len(path)
                return
            for v in self.token_tree[k]:
                path.append(v)
                dfs(v, path)
                path.pop()

        dfs(0, [0])

        # Need to pad and also collect the mask
        mask = torch.zeros((len(paths), max_len), dtype=torch.int64)
        all_paths = torch.full((len(paths), max_len), self.k-1, dtype=torch.int64)
        max_accepted = torch.zeros((len(paths), 1), dtype=torch.int16)

        for i in range(len(paths)):
            for j in range(len(paths[i])):
                all_paths[i][j] = paths[i][j]
                mask[i][j] = 1
                max_accepted[i] += 1

        return all_paths, mask, max_accepted

    def _reconstruct_target_ids(self, target_ids, orig_target_ids):
        """
        Helper function for combining target_ids sampled from original distribution
        and adjusted distribution
        """
        new_target_ids = torch.zeros_like(target_ids)

        def dfs(k):
            nonlocal new_target_ids

            if k not in self.token_tree.keys():
                new_target_ids[:, k] = orig_target_ids[:, k]
                return
            new_target_ids[:, k] = target_ids[:, k]
            for v in self.token_tree[k]:
                dfs(v)

        dfs(0)

        return new_target_ids

    def _get_cache_update_indices(self, cache_ids, all_paths, index):
        """
        Helper function for getting the cache update indices given the accepted path
        gather_indices = accepted cache ids from the previous iteration
        scatter_indices = scatter the accepted cache ids to the right cache postion
        """
        batch_size = cache_ids.shape[0]
        offsets = cache_ids.clone()[:, :1]
        scatter = torch.arange(0, self.depth)
        all_paths = all_paths.reshape(1, -1, self.depth).expand(batch_size, -1, -1)
        gather = torch.gather(all_paths, 1, index).reshape(batch_size, self.depth)

        cache_gather_indices = offsets + gather
        cache_scatter_indices = offsets + scatter

        return cache_gather_indices, cache_scatter_indices

    def hlo(self, batch_size=1, n_positions=128):

        draft = self.draft.decoder_lm_head
        target = self.target.decoder_lm_head

        draft.builder.n_positions = n_positions
        target.builder.n_positions = n_positions
        draft.builder.n_active_tokens = self.k
        target.builder.n_active_tokens = self.k

        num_inputs = 0
        num_outputs = 0

        def speculator(scribe):
            amp, *_ = utils.parse_amp(draft.amp)
            dtype = getattr(scribe, amp)
            s32 = scribe.s32

            debugs = []

            # Allocate global inputs
            (tokens,
             cache_ids,
             start_ids,
             last_token_id,
             block_tables,
             context_lens,
             prev_hidden,
             tree_mask,
             update_indices,
             hidden_update_indices,
             cache_gather_indices,
             cache_scatter_indices,
             position_ids,
             all_paths), inputs_sdim = draft.builder.eagle_draft_inputs(
                scribe, dtype, 1, batch_size, token_tree=True, k=self.k, n_leaves=self.n_leaves, depth=self.depth, n_entrees=self.tree_matrix.shape[0], width=self.width
            )

            nonlocal num_inputs
            num_inputs = len(inputs_sdim)

            # Allocate Parameters for weight/cache tensors
            param_builder = decoder.DecoderParameterBuilder(scribe, num_inputs)
            draft_in_caches, draft_layers_weights, draft_pre_layer_params, draft_lm_head_params, draft_generation_params = draft._hlo_parameters(n_positions, batch_size, param_builder)

            # Reorder KV cache based on previous acceptance results
            caches = []
            cache = draft_in_caches[0][0]
            # TODO: Add check for BSH layout
            _, _, n_kv_heads, d_head = cache.sizes
            draft_cache_gather_indices = hlo.transpose(cache_gather_indices, 0, 1)
            draft_cache_gather_indices = hlo.reshape(draft_cache_gather_indices, [self.depth, batch_size, 1, 1])
            draft_cache_gather_indices = hlo.broadcast(draft_cache_gather_indices, [self.depth, batch_size, n_kv_heads, d_head], [0, 1, 2, 3])
            for layer_cache in draft_in_caches:
                k_cache, v_cache = layer_cache
                draft_cache_keys = hlo.gather(k_cache,0, draft_cache_gather_indices)
                draft_cache_values = hlo.gather(v_cache, 0, draft_cache_gather_indices)
                k_cache, v_cache = fused_kv_update_cache(k_cache, v_cache, cache_scatter_indices, draft_cache_keys, draft_cache_values, start_ids, self.neuron_config)
                caches.append([k_cache, v_cache])

            # Create lists to aggregate outputs (used for token acceptance)
            draft_logits = list()
            draft_tokens = [tokens]

            # Prapare inputs for draft speculation
            orig_tokens = tokens
            orig_hidden = prev_hidden
            tokens = hlo.broadcast(tokens, [batch_size, self.k], [0, 1])
            _, _, hidden_size = prev_hidden.sizes
            prev_hidden = hlo.broadcast(prev_hidden, [batch_size, self.k, hidden_size], [0, 1, 2])
            # TODO: Cache id should be built by input
            prior_ids = [cache_ids]
            for _ in range(self.k-1):
                cache_ids = hlo.add(cache_ids, 1)
                prior_ids.append(cache_ids)
            draft_cache_ids = hlo.concatenate(prior_ids, 1)

            target_cache_ids = hlo.concatenate(prior_ids, 1)
            target_cache_ids = hlo.add(target_cache_ids, 1)

            cache_ids = draft_cache_ids

            hidden_update_indices = hlo.reshape(hidden_update_indices, [batch_size, self.k-1, 1])
            hidden_update_indices = hlo.broadcast(hidden_update_indices, [batch_size, self.k-1, hidden_size], [0, 1, 2])

            for _ in range(self.depth-1):
                tensors = draft_cache_ids, start_ids, last_token_id, block_tables, context_lens, prev_hidden
                logits, new_hidden, caches = draft._hlo_eagle_draft_unroll(tokens, tensors, caches, draft_layers_weights, draft_pre_layer_params, draft_lm_head_params, tree_mask, position_ids)
                logits = hlo.permute(logits, (2, 1, 0))
                _, new_tokens = hlo.topk(logits, k=self.width, dim=2, tp_degree=self.tp_degree)
                new_tokens = hlo.reshape(new_tokens, [batch_size, self.k * self.width])
                new_tokens = hlo.cast(new_tokens, s32)
                new_tokens = hlo.gather(new_tokens, 1, update_indices)
                new_hidden = hlo.gather(new_hidden, 1, hidden_update_indices)
                tokens = hlo.concatenate([orig_tokens, new_tokens], 1)
                prev_hidden = hlo.concatenate([orig_hidden, new_hidden], 1)

            # Since we are using greedy for draft, we directly set the probs to 1 for draft
            draft_probs = hlo.full(1.0, logits.dtype, [batch_size, self.k-1, 1])
            draft_indices = hlo.reshape(new_tokens, [batch_size, self.k-1, 1])

            draft_caches = caches
            draft_position_ids = position_ids

            # Concatenate all draft outputs
            cache_ids = target_cache_ids

            # Execute target model
            in_caches, layers_weights, pre_layer_params, lm_head_params, generation_params = target._hlo_parameters(n_positions, batch_size, param_builder)
            caches = []
            cache = in_caches[0][0]
            _, _, n_kv_heads, d_head = cache.sizes
            # TODO: we probably don't need to do these again
            cache_scatter_indices = hlo.add(cache_scatter_indices, 1)
            cache_gather_indices = hlo.add(cache_gather_indices, 1)
            target_cache_gather_indices = hlo.transpose(cache_gather_indices, 0, 1)
            target_cache_gather_indices = hlo.reshape(target_cache_gather_indices, [self.depth, batch_size, 1, 1])
            target_cache_gather_indices = hlo.broadcast(target_cache_gather_indices, [self.depth, batch_size, n_kv_heads, d_head], [0, 1, 2, 3])
            #target_cache_gather_indices = hlo.add(target_cache_gather_indices, 1)
            for layer_cache in in_caches:
                k_cache, v_cache = layer_cache
                # We only support BSH
                target_cache_keys = hlo.gather(k_cache,0, target_cache_gather_indices)
                target_cache_values = hlo.gather(v_cache, 0, target_cache_gather_indices)
                k_cache, v_cache = fused_kv_update_cache(k_cache, v_cache, cache_scatter_indices, target_cache_keys, target_cache_values, start_ids, self.neuron_config)
                caches.append([k_cache, v_cache])

            tensors = target_cache_ids, start_ids, last_token_id, block_tables, context_lens
            position_ids = hlo.add(position_ids, 1)
            target_scores, hidden, caches = target._hlo_eagle_target_unroll(tokens, tensors, caches, layers_weights, pre_layer_params, lm_head_params, tree_mask, position_ids)
            target._hlo_cache_aliases(in_caches, caches)

            # Execute final iteration in case all tokens are accepted.
            # Need to use target's hidden to update the draft cache
            new_hidden = hlo.gather(hidden, 1, hidden_update_indices)
            prev_hidden = hlo.concatenate([orig_hidden, new_hidden], 1)
            tensors = draft_cache_ids, start_ids, last_token_id, block_tables, context_lens, prev_hidden
            logits, _, draft_caches = draft._hlo_eagle_draft_unroll(tokens, tensors, draft_caches, draft_layers_weights, draft_pre_layer_params, draft_lm_head_params, tree_mask, draft_position_ids)
            draft_scores = logits
            draft._hlo_cache_aliases(draft_in_caches, draft_caches)

            # Prepare output caches
            draft_out_caches = itertools.chain(*draft_caches)
            target_out_caches = itertools.chain(*caches)

            # Sample output tokens
            target_probs, target_indices = target._hlo_generation(target_scores, generation_params, early_return=True)
            orig_indices = target_indices
            orig_probs = target_probs

            # Rebase the target probabilities in case of rejection
            # We first collect all possible paths for draft and targets
            draft_ids = hlo.broadcast(tokens, [batch_size, self.n_leaves, self.k], [0, 2])
            partial_all_paths = hlo.slice_along(all_paths, 1, limit=self.depth, start=1)
            partial_all_paths = hlo.broadcast(partial_all_paths, [batch_size, self.n_leaves, self.depth-1], [1, 2])

            draft_ids = hlo.gather(draft_ids, 2, partial_all_paths)
            draft_indices = hlo.reshape(draft_ids, [batch_size * self.n_leaves, self.depth-1, 1])
            draft_probs = hlo.full(1.0, logits.dtype, [batch_size * self.n_leaves, self.depth-1, 1])

            _, _, vocab_size = target_indices.sizes
            target_indices = hlo.broadcast(target_indices, [batch_size, self.n_leaves, self.k, vocab_size], [0, 2, 3])
            all_paths = hlo.broadcast(all_paths, [batch_size, self.n_leaves, self.depth, vocab_size], [1, 2])
            target_indices = hlo.gather(target_indices, 2, all_paths)
            target_indices = hlo.reshape(target_indices, [batch_size * self.n_leaves, self.depth, vocab_size])
            target_indices = hlo.cast(target_indices, draft_indices.dtype)

            target_probs = hlo.broadcast(target_probs, [batch_size, self.n_leaves, self.k, vocab_size], [0, 2, 3])
            target_probs = hlo.gather(target_probs, 2, all_paths)
            target_probs = hlo.reshape(target_probs, [batch_size * self.n_leaves, self.depth, vocab_size])

            # Then we get the adjusted probs for each possible path
            adjusted_target_probs = hlo.speculative_adjust_distribution(
                    draft_probs,
                    draft_indices,
                    target_probs,
                    target_indices,
                    self.depth-1)

            generation_config = target.neuron_config.on_device_generation
            target_ids = hlo.multinomial(adjusted_target_probs, dim=2, deterministic=generation_config.deterministic)
            target_ids = hlo.gather(target_indices, 2, target_ids)
            target_ids = hlo.squeeze(target_ids, 2)
            target_ids = hlo.cast(target_ids, target_ids.scribe.s32)

            draft_ids = hlo.reshape(draft_ids, [batch_size * self.n_leaves, self.depth-1])

            sliced_target_indices = hlo.slice_along(target_indices, 1, limit=self.depth-1)
            sliced_target_probs = hlo.slice_along(target_probs, 1, limit=self.depth-1)

            next_tokens, index, *mask = hlo.speculative_token_selection(
                draft_ids, target_ids,
                draft_indices, draft_probs,
                sliced_target_indices, sliced_target_probs,
                tp_degree=self.tp_degree,
                pad_token_id=self.pad_token_id,
                deterministic_threshold=None, output_mask=self.output_scores
            )

            next_tokens = hlo.reshape(next_tokens, [batch_size, self.n_leaves, self.depth])
            index = hlo.reshape(index, [batch_size, self.n_leaves])

            counts = hlo.add(index, 1)

            outputs = [
                next_tokens,
                counts,
                hidden,
            ]

            # Retrieve the target model output scores for the selected tokens
            target_output_scores = None
            if self.output_scores:
                mask, = mask
                target_scores = hlo.all_gather(target_scores, dim=0, tp_degree=self.tp_degree) # Collect scores from all ranks
                target_scores = hlo.permute(target_scores, (2, 1, 0)) # (vocab, k + 1, batch_size) -> (batch_size, k + 1, vocab)
                mask = hlo.broadcast(mask, target_scores.sizes, [0, 1]) # (batch_size, k + 1) -> (batch_size, k + 1, vocab)
                target_output_scores = hlo.masked_select(mask, target_scores, float('-inf'))

            outputs = outputs + debugs
            if self.output_scores:
                outputs.append(target_output_scores)
            nonlocal num_outputs
            num_outputs = len(outputs)
            outputs = outputs + [*draft_out_caches, *target_out_caches]
            outputs = [out for out in outputs if out is not None]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        hlo_module = compiler.compile_py_func(speculator)
        return (
            hlo_module,
            num_inputs,
            num_outputs,
        )

    def to_neuron(self, workers=None):

        sizes = list(itertools.product(self.batch_sizes, self.buckets))

        hlo_modules = list()
        for (batch_size, sequence_length) in sizes:
            hlo_module, num_inputs, num_outputs = self.hlo(batch_size, sequence_length)
            hlo_modules.append(hlo_module)

        selector = program.FusedSpeculativeSelector(sizes, self.k)
        self.speculator = program.BucketedParallelProgram(
            hlo_modules,
            selector=selector,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            neuron_config=self.neuron_config,
            tp_degree=self.tp_degree,
            tags=[
                f'fused-speculator-seqlen{sequence_length}-batch{batch_size}'
                for batch_size, sequence_length in sizes
            ],
        )
        draft_params = self.draft.decoder_lm_head.valid_parameters(sequence_length, batch_size)
        target_params = self.target.decoder_lm_head.valid_parameters(sequence_length, batch_size)
        self.speculator.build(workers)
        self.speculator.setup([*draft_params, *target_params])

    def update_generation_config(self, generation_config: GenerationConfig):
        self.draft.update_generation_config(generation_config)
        self.target.update_generation_config(generation_config)

    def _sample_loop(
        self,
        batch_size,
        token_id,
        cache_ids,
        start_ids,
        hidden,
        tree_mask,
        update_indices,
        hidden_update_indices,
        cache_gather_indices,
        cache_scatter_indices,
        all_paths,
        max_accepted,
    ):
        last_token_id = torch.as_tensor([0], dtype=torch.int32).expand(batch_size)
        position_ids = self._get_position_ids(batch_size, cache_ids)
        block_tables = torch.as_tensor([0])
        context_lens = torch.as_tensor([0])

        if self.debug:
            print("speculator inputs:",
                    "full_input_ids", token_id,
                    "cache_ids_pad", cache_ids,
                    "seq_ids_pad", start_ids,
                    "hidden_pad", hidden[:, :, 10])

        outputs = self.speculator.execute(
                token_id,
                cache_ids,
                start_ids,
                last_token_id,
                block_tables,
                context_lens,
                hidden,
                tree_mask,
                update_indices,
                hidden_update_indices,
                cache_gather_indices,
                cache_scatter_indices,
                position_ids,
                all_paths,
                return_ranks=1)

        next_tokens, counts, hidden, *debugs = outputs




        # Post-processing for preparing the next iteration
        # We all the paths and the right tokens, we find the longest accepted path
        # If two paths have the same token counts, we accept from the highest probabilites
        # (Note that probabilites are already sorted because of topk)
        counts = torch.clamp(counts, max=max_accepted)
        max_path = torch.argmax(counts, dim=1, keepdim=True)
        counts = torch.gather(counts, 1, max_path)
        max_path = torch.reshape(max_path, (batch_size, -1, 1)).expand(-1, -1, next_tokens.shape[-1])
        tokens = torch.gather(next_tokens, 1, max_path)
        tokens = torch.reshape(tokens, (batch_size, -1))

        # Post-processing for preparing the next iteration
        # Get the last token, which becomes the input to the next iteration
        token_id = torch.gather(tokens, 1, counts.to(torch.int64)-1)


        if self.debug:
            print("speculator outputs:",
                    "next_tokens", next_tokens,
                    "counts", counts,
                    "token_id", token_id,
                    "next_hidden", hidden[:, :, 10])

        # Get the new ordering of the cache ids
        cache_gather_indices, cache_scatter_indices = self._get_cache_update_indices(cache_ids, all_paths, max_path)

        # Ger the correct hidden value according to the accepted token
        _all_paths = all_paths.reshape(1, -1, self.depth).expand(batch_size, -1, -1)
        path = torch.gather(_all_paths, 1, max_path)
        _counts = counts.to(torch.int64).reshape(batch_size, 1, -1) - 1
        path = torch.gather(path, -1, _counts)
        path = path.expand(-1, -1, hidden.shape[-1])
        hidden = torch.gather(hidden, 1, path)

        return token_id, cache_gather_indices, cache_scatter_indices, hidden, tokens, counts

    def sample(
        self,
        input_ids: torch.Tensor,
        start_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None,
        streamer: Optional['transformers.generation.streamers.BaseStreamer'] = None,
    ):
        """
        Sample tokens using the fully fused speculative graph.

        Args:
            input_ids: The tokenized input identifiers.
            start_ids: The offset from the beginning of each input in a batch.
                For batched speculation, this is sequence ids.
            attention_mask: A mask which indicates which input tokens
                should be attended to.
            sequence_length: The total length of inputs + outputs.
            streamer: A streamer callback object for generated tokens. During
                execution, the sampling loop will stream full batches to the
                `streamer` object with padding included. The value will be the
                `pad_token_id` provided at construction. The streamer should
                handle special tokens to eliminate these identifiers.

        Returns:
            tokens: The generated sequences.
            scores: (optional) The target model scores if output_scores is set.

        Examples:
            draft = NeuronAutoModelForCausalLM.from_pretrained(...)
            target = NeuronAutoModelForCausalLM.from_pretrained(...)

            draft.to_neuron()
            target.to_neuron()

            fsd = FusedSpeculativeDecoder(draft, target, 5)
            fsd.to_neuron()

            fsd.sample(input_ids, sequence_length=256)
        """
        # FIXME: Handle attention_mask. Currently, attention_mask is not used.

        if sequence_length is None:
            sequence_length = self.max_position - self.k # type: int
        # FIXME: Loosen this restriction to sequence_length <= max_position
        assert sequence_length <= self.max_position - self.k

        batch_size, start = input_ids.shape
        if start_ids is None:
            start_ids = torch.arange(batch_size)

        draft_attention_mask = None
        assert attention_mask is not None, (
            "The attention mask needs to be provided for speculation where batch_size>1"
        )
        draft_attention_mask = attention_mask[:, 1:]
        cache_ids = torch.arange(start).reshape(1, start).expand(batch_size, start)
        draft_cache_ids = torch.arange(start-1).reshape(1, start-1).expand(batch_size, start-1).mul(draft_attention_mask)

        # The streamer should send back the input tokens to conform to
        # huggingface behavior.
        # Reference: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/generation/utils.py#L1410-L1411
        if streamer:
            streamer.put(input_ids)

        # Context encoding
        # FIXME: populate scores with the context encoding logits


        token_id, hidden = self.target(input_ids, cache_ids=cache_ids, start_ids=start_ids)

        if streamer:
            streamer.put(token_id)

        if self.debug:
            print("draft ctx inputs:",
                    "token_id", input_ids[:,1:],
                    "cache_ids", draft_cache_ids,
                    "start_ids", start_ids,
                    "hidden", hidden[:, :, 10])

        outputs = self.draft(input_ids[:,1:], cache_ids=draft_cache_ids, start_ids=start_ids, prev_hidden=hidden)

        # Preallocate state tensors
        sequences = torch.full((batch_size, sequence_length + self.k + 1), self.pad_token_id, dtype=torch.int32)
        sequences[:, :start] = input_ids
        cache_ids = torch.count_nonzero(attention_mask,dim=1).view(-1, 1) - 1
        positions = torch.count_nonzero(attention_mask,dim=1).view(-1, 1) + torch.arange(self.depth, dtype=torch.int64).repeat((batch_size, 1))
        sequences.scatter_(1, torch.count_nonzero(attention_mask,dim=1).view(-1, 1), token_id)
        positions = positions + 1
        gather_index = cache_ids.reshape(batch_size, -1, 1).expand(-1, -1, hidden.shape[-1])
        hidden = torch.gather(hidden, 1, gather_index)

        if self.output_scores:
            # We cut off the first `start` tokens when returning scores
            scores = torch.full((list(sequences.shape) + [self.vocab_size]), self.pad_token_id, dtype=torch.float32)

        # A tensor which keeps track of which sequences are done
        done = torch.full((batch_size, 1), False)

        # A tensor which keeps track of the sequence ends
        ends = torch.full((batch_size,), sequence_length, dtype=torch.int32)

        # Tensors to track local batch positions/masking for early stop
        batch_positions = torch.arange(self.depth).unsqueeze(0)
        batch_mask = torch.full((batch_size, self.depth), False)

        # Minor optimization: Convert token types to tensors to avoid casting
        eos_token = None
        if self.eos_token_id is not None:
            eos_token = torch.tensor(self.eos_token_id, dtype=torch.int32)
        sequence_length = torch.tensor(sequence_length, dtype=torch.int32)

        accepts = 0
        iters = torch.zeros(batch_size,)
        rate = torch.zeros(batch_size, 10)

        tree_mask = generate_attention_mask(self.token_tree)
        update_indices = self._get_update_indices(batch_size)
        hidden_update_indices = self._get_hidden_update_indices(batch_size)
        all_paths, token_mask, max_accepted = self._get_all_paths()
        max_accepted = max_accepted.reshape(1, self.n_leaves).expand(batch_size, -1)

        cache_gather_indices = torch.arange(0, self.depth).expand(batch_size, self.depth).to(torch.int64)
        cache_scatter_indices = torch.arange(0, self.depth).expand(batch_size, self.depth).to(torch.int64)

        while True:

            token_id, cache_gather_indices, cache_scatter_indices, hidden, tokens, counts = self._sample_loop(
                batch_size,
                token_id,
                cache_ids,
                start_ids,
                hidden,
                tree_mask,
                update_indices,
                hidden_update_indices,
                cache_gather_indices,
                cache_scatter_indices,
                all_paths,
                max_accepted,
            )

            for batch in range(batch_size):
                if not done[batch]:
                    accepts += counts
                    iters[batch] += 1
                    rate[batch][counts[0][0]] += 1

            if eos_token is not None:

                # Do a minimal check for the eos_token to keep the happy-path fast
                # We only need to check once per sequence
                finished = []
                for batch in range(batch_size):
                    for t in range(counts[batch][0]):
                        if tokens[batch][t] == eos_token:
                            finished.append([batch, t])
                finished = torch.tensor(finished)
                #finished = torch.nonzero(torch.eq(tokens, eos_token).logical_and_(torch.logical_not(done)))
                if finished.numel():
                    sequence, position = finished[:, 0], finished[:, 1]
                    for seq, pos in zip(sequence, position):
                        ends[seq] = cache_ids[seq][0].int() + pos.int() + 2
                        done[seq] = True
                        batch_mask[seq] = torch.greater(batch_positions, pos)

                # Always fill tokens after the eos_token with the pad_token.
                # This needs to be done prior to the streamer call to avoid
                # streaming back outputs beyond the stop token.
                tokens.masked_fill_(batch_mask, self.pad_token_id)

                # If a stop token was just found, set future batches to be padded
                if finished.numel():
                    batch_mask.logical_or_(done)

            if streamer:
                streamer.put(tokens)

            sequences.scatter_(1, positions, tokens)
            if self.output_scores:
                score, = score
                # TODO: Look into the perf benefit of performing concats instead
                scores_positions = torch.broadcast_to(torch.unsqueeze(positions, -1), list(positions.shape) + [self.vocab_size])
                scores.scatter_(1, scores_positions, score)
            positions += counts
            cache_ids += counts

            # Clamp the cache_ids to the sequence length so that any batch lines
            # that have not reached the end can continue. For batch lines that
            # are complete this populates garbage data into KV cache tail beyond
            # the sequence_length.
            positions = torch.clamp(positions, max=sequence_length)
            cache_ids = torch.clamp(cache_ids, max=sequence_length)
            done.logical_or_(torch.eq(cache_ids, sequence_length))

            if done.all().item():
                break

        if streamer:
            streamer.end()

        if self.output_scores:
            return sequences[:, :ends.max().item()], scores[:, start:ends.max().item(), :]
        return sequences[:, :ends.max().item()], rate, iters

    def speculative_iteration(
            self,
            input_ids: torch.Tensor,
            cache_ids: torch.Tensor,
            start_ids: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):

        """
        eagle speculative iteration for continous batching.
        Note that this method only do one speculative iteration.
        user need to define the sampling loop for the whole sequence generation.

        Args:
            input_ids:
                The input token ids passed to the model to generate
                next predicted tokens (sequence_length - len(input_ids)).
            cache_ids:
                The positions in the KV cache that should be updated.
                shape=(batch, seqlen) for continous batching
            start_ids:
                The offset from the beginning of each input in a batch.

        Returns:
            tokens (list of [tensor of shape (accepted_token_length)]):
        """
        batch_size, context_len = input_ids.shape
        cache_ids = self.prepare_cache_ids(cache_ids, batch_size, context_len)
        min_values_per_row, _ = torch.min(cache_ids, dim=-1)

        # in continuous batching, we need to identify new request and do context decoding
        do_context_decoding = (min_values_per_row == 0).any()
        if do_context_decoding:
            if self.debug:
                print("target ctx inputs:",
                        "token_id", input_ids,
                        "cache_ids", cache_ids,
                        "start_ids", start_ids)

            target_next_id, hidden = self.target(input_ids, cache_ids, start_ids)

            if self.debug:
                    print("draft ctx inputs:",
                            "token_id", input_ids[:,1:],
                            "cache_ids", cache_ids,
                            "start_ids", start_ids,
                            "hidden", hidden[:, :, 10])

            outputs = self.draft(input_ids[:,1:], cache_ids=cache_ids[:,:-1], start_ids=start_ids, prev_hidden=hidden)
            next_cache_ids = torch.count_nonzero(cache_ids, dim=1).view(-1, 1)
            gather_index = next_cache_ids.reshape(batch_size, -1, 1).expand(-1, -1, hidden.shape[-1])
            hidden = torch.gather(hidden, 1, gather_index)
            if self.debug:
                print("hidden_gather_indx", gather_index[:, :, 0])
            ctr = 0
            for i in start_ids:
                self.hidden[i.item()]=hidden[ctr].unsqueeze(0) if hidden.shape[0] > 1 else hidden
                ctr = ctr + 1
                self.cache_gather_indices[i.item()] = torch.arange(0, self.depth).unsqueeze(0).to(torch.int64)
                self.cache_scatter_indices[i.item()] = torch.arange(0, self.depth).unsqueeze(0).to(torch.int64)
            return target_next_id, torch.tensor([[1]] * batch_size)

        seq_ids = start_ids
        # TODO: enable multiple bucket in batch dimension
        graph_batch_size = self.batch_sizes[0]
        indv_hidden = [self.hidden[i.item()] for i in seq_ids]
        indv_cache_gather_indices = [self.cache_gather_indices[i.item()] for i in seq_ids]
        indv_cache_scatter_indices = [self.cache_scatter_indices[i.item()] for i in seq_ids]
        hidden = torch.cat(indv_hidden, dim=0)
        cache_gather_indices = torch.cat(indv_cache_gather_indices, dim=0)
        cache_scatter_indices = torch.cat(indv_cache_scatter_indices, dim=0)
        cache_ids = cache_ids - 1
        # full_input_ids, cache_ids_pad, hidden_pad, cache_gather_indices_pad, cache_scatter_indices_pad = self.handle_padding_list(
        #     [input_ids, cache_ids, hidden, cache_gather_indices, cache_scatter_indices],
        #     seq_ids,
        #     graph_batch_size
        # )

        def pad(tensor, seq_ids, target_batch):

            padded_tensor = torch.zeros([target_batch] + list(tensor.shape[1:]), dtype=tensor.dtype)

            padded_tensor[seq_ids] = tensor

            return padded_tensor

        full_input_ids, cache_ids_pad, hidden_pad, cache_gather_indices_pad, cache_scatter_indices_pad = [
            pad(t, seq_ids, graph_batch_size)
            for t in [input_ids, cache_ids, hidden, cache_gather_indices, cache_scatter_indices]
        ]

        seq_ids_pad = torch.arange(graph_batch_size)

        tree_mask = generate_attention_mask(self.token_tree)
        update_indices = self._get_update_indices(graph_batch_size)
        hidden_update_indices = self._get_hidden_update_indices(graph_batch_size)
        all_paths, token_mask, max_accepted = self._get_all_paths()
        max_accepted = max_accepted.reshape(1, self.n_leaves).expand(graph_batch_size, -1)

        token_id, cache_gather_indices, cache_scatter_indices, hidden, tokens, counts = self._sample_loop(
            graph_batch_size,
            full_input_ids,
            cache_ids_pad,
            seq_ids_pad,
            hidden_pad,
            tree_mask,
            update_indices,
            hidden_update_indices,
            cache_gather_indices_pad,
            cache_scatter_indices_pad,
            all_paths,
            max_accepted,
        )

        output_tokens, output_counts, output_hidden, output_cache_gather_indices, output_cache_scatter_indices = self.handle_padding_list(
            [tokens, counts, hidden, cache_gather_indices, cache_scatter_indices],
            seq_ids, batch_size)
        ctr = 0
        for i in start_ids:
            self.hidden[i.item()]=torch.unsqueeze(output_hidden[ctr], 0)
            self.cache_gather_indices[i.item()]=torch.unsqueeze(output_cache_gather_indices[ctr], 0)
            self.cache_scatter_indices[i.item()]=torch.unsqueeze(output_cache_scatter_indices[ctr], 0)
            ctr = ctr + 1

        return self.speculative_iteration_post_process(output_tokens, output_counts)


    def handle_padding_list(
            self,
            tensor_list: List[torch.Tensor],
            seq_ids: torch.Tensor,
            target_batch_size: int
    ) -> List[torch.Tensor]:
        return [self.handle_padding(tensor, seq_ids, target_batch_size) for tensor in tensor_list]

    def handle_padding(
            self,
            tensor: torch.Tensor,
            seq_ids: torch.Tensor,
            target_batch_size: int
    ) -> torch.Tensor:
        """
         add or remove padding for when running batch size is different from target_batch_size(neff batch size)
         i.e. input_tensor [[2],[3]], seq_ids: [2, 1] target_batch_size:3 -> [[0], [2], [3]]
              input_tensor [2, 1]   , seq_ids: [2, 1] target_batch_size:3-> [0, 2, 1]
              input_tensor [[1,2,3,4],[11,12,13,14],[21,22,23,24]], seq_ids: [0, 2, 1] taget_batch_size:2
              ->[[11,12,13,14],[21,22,23,24]]
            input_tensor [6,6,6], seq_ids: [0, 2, 1] target_batch_size:2 ->[6,6]

         Args:
             tensor:
                 The input tensor to be processed. first dimension is batch. dim_size >=1
             seq_ids:
                 The positions in the KV cache that should be updated.
             target_batch_size:
                 batch size the output tensor should have

         Returns:
             tensors, first dimension is the target batch size.
         """

        original_shape = tensor.shape
        current_batch_size = original_shape[0]

        if current_batch_size == target_batch_size:
            return tensor

        new_shape = (target_batch_size,) + original_shape[1:]
        output_tensor = torch.zeros(new_shape, dtype=tensor.dtype)
        for idx, seq_id in enumerate(seq_ids):
            if current_batch_size < target_batch_size:
                output_tensor[seq_id] = tensor[idx]
            else:
                output_tensor[idx] = tensor[seq_id]
        return output_tensor

    def prepare_cache_ids(self, cache_ids: torch.Tensor, batch: int, seq_len: int) -> List[torch.Tensor]:
        """
        Args:
            cache_ids:
                The positions in the KV cache that should be updated.
            batch:
                The batch size
            seq_len:
                sequence length

        Returns:
            cache_ids of shape=(batch, seqlen) for continous batching or (seq_len,) for non-continous batching
        """
        if cache_ids is not None:
            return cache_ids
        if self.target.neuron_config and self.target.neuron_config.use_2d_cache_ids:
            return torch.tensor([[j for j in range(seq_len)] for i in range(batch)])
        else:
            return torch.tensor([i for i in range(seq_len)])
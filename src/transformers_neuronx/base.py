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
import pickle

# Mainly used to expose top level APIs to the model object for serialization
class NeuronModelBase:

    # top level api
    def save(self, directory):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._save_compiled_artifacts(directory)

    # top level api
    def load(self, directory):
        assert self.serialization_enabled(), 'serialization is not enabled for this model'
        self._load_compiled_artifacts(directory)

    def reorder_cache(self, reorder_ids):
        self.decoder_lm_head.program.reorder_cache(reorder_ids)

    def setup_reorder_cache(self):
        if self.decoder_lm_head.program is not None: # called after to_neuron
            self.decoder_lm_head.program.setup_reorder_cache()
        else:
            self.decoder_lm_head.need_reorder_cache = True

    def _save_compiled_artifacts(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f'Artifacts should be saved to a directory. '
                f'Found existing file: {directory}'
            )
        os.makedirs(directory, exist_ok=True)
        for i, nbs_obj in enumerate(self.nbs_objs):
            nbs_obj.save_compiler_artifacts(os.path.join(directory, f'neuron-program-{i}.pkl'))

    def _load_compiled_artifacts(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'Did not find directory: {directory}, '
                                    f'ensure that your saved model and loaded model are using the same parameters.')

        for i, nbs_obj in enumerate(self.nbs_objs):
            program_filename = os.path.join(directory, f'neuron-program-{i}.pkl')
            nbs_obj.load_compiler_artifacts_after_build(program_filename)

    # To enable serialization, have the model call this 
    # function to register all nbs_obj of your model.
    # The nbs_obj must follow 3 rules:
    #   1. The nbs_obj must inherit from NeuronBaseSerializer.
    #   2. Since this class shouldn't be used directly, a nbs_obj.get_all_kernels()
    #      method should be implemented by the child class, which returns a
    #      list of all kernels which have NEFFs.
    #   3. It must use:
    #      if nbs_obj.compiler_artifacts_path is not None:
    #          nbs_obj.set_neff_bytes()
    #      after its kernels have been created, but before they are compiled
    def register_for_serialization(self, nbs_obj):
        # check that at least requirement 1 and 2 are met, 3 is hard to check for here
        assert issubclass(type(nbs_obj), NeuronBaseSerializer), 'The nbs_obj must inheret from NeuronBaseSerializer.'
        assert getattr(nbs_obj, 'get_all_kernels', None) is not None, 'An nbs_obj.get_all_kernels() method should be implemented.'
        temp = getattr(self, 'nbs_objs', [])
        nbs_obj.compiler_artifacts_path = None
        temp.append(nbs_obj)
        self.nbs_objs = temp

    def serialization_enabled(self):
        return getattr(self, 'nbs_objs', None) is not None

# Base class for all "Serializable Objects"
class NeuronBaseSerializer:

    def save_compiler_artifacts(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_neff_bytes(), f)
    
    def load_compiler_artifacts_after_build(self, path):
        self.compiler_artifacts_path = path
    
    def get_neff_bytes(self):
        neff_bytes_arr = [kernel.neff_bytes for kernel in self.get_all_kernels()]
        return neff_bytes_arr

    def set_neff_bytes(self):
        with open(self.compiler_artifacts_path, 'rb') as f:
            kernels_neff_bytes = pickle.load(f)    
        for kernel, neff_bytes in zip(self.get_all_kernels(), kernels_neff_bytes):
            kernel.neff_bytes = neff_bytes

    def get_all_kernels(self):
        raise NotImplementedError(
            f'Class {type(self)} deriving from NeuronBaseSerializer must implement get_all_kernels'
        )

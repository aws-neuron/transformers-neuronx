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

class NeuronModelBase:

    def reorder_cache(self, reorder_ids):
        self.decoder_lm_head.reorder_cache(reorder_ids)

    def setup_reorder_cache(self):
        self.decoder_lm_head.setup_reorder_cache()

    def _save_compiled_artifacts(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(
                f'Artifacts should be saved to a directory. '
                f'Found existing file: {directory}'
            )
        os.makedirs(directory, exist_ok=True)
        self.decoder_lm_head.save_compiler_artifacts(os.path.join(directory, 'neuron-program.pkl'))

    def _load_compiled_artifacts(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'Did not find directory: {directory}')
        program_filename = os.path.join(directory, 'neuron-program.pkl')
        if os.path.exists(program_filename):
            self.decoder_lm_head.load_compiler_artifacts_after_build(program_filename)

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation marks.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="punct", kind="classify", deterministic=deterministic)

        # Use pynini.accep() for characters that might be interpreted as regex
        punct_chars = [
            ".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")",
            "/", "।", "॥",  # Kannada/Devanagari danda and double danda
        ]
        # Build union using accep() to avoid regex interpretation
        punctuation = pynini.union(*[pynini.accep(c) for c in punct_chars])

        graph = pynutil.insert("name: \"") + punctuation + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()

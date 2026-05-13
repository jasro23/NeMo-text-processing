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

from nemo_text_processing.text_normalization.kn.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space
from nemo_text_processing.text_normalization.kn.taggers.decimal import quantities

# Minus prefix without leading space for verbalization output
MINUS_PREFIX = pynini.union("ಋಣಾತ್ಮಕ ", "ಮೈನಸ್ ").optimize()


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal numbers, e.g.
        decimal { negative: "true" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐದು ಸೊನ್ನೆ ಸೊನ್ನೆ ಆರು" } -> ಮೈನಸ್ ಹನ್ನೆರಡು ದಶಮಾಂಶ ಐದು ಸೊನ್ನೆ ಸೊನ್ನೆ ಆರು
        decimal { integer_part: "ಮೂರು" fractional_part: "ಒಂದು ನಾಲ್ಕು" } -> ಮೂರು ದಶಮಾಂಶ ಒಂದು ನಾಲ್ಕು
        decimal { integer_part: "ಒಂದು" quantity: "ಲಕ್ಷ" } -> ಒಂದು ಲಕ್ಷ

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        delete_space = pynutil.delete(" ")
        
        # Handle negative sign (no leading space in output)
        self.optional_sign = pynini.closure(pynini.cross("negative: \"true\"", MINUS_PREFIX) + delete_space, 0, 1)
        
        # Integer part
        self.integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        
        # Fractional part (digit by digit)
        self.fractional_default = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        # Insert "ದಶಮಾಂಶ" (decimal point) before fractional part
        self.fractional = pynutil.insert(" ದಶಮಾಂಶ ") + self.fractional_default

        # Quantity (lakh, crore, etc.)
        self.quantity = (
            delete_space + insert_space + pynutil.delete("quantity: \"") + quantities + pynutil.delete("\"")
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        # Graph composition:
        # 1. integer + quantity (e.g., "1 lakh")
        # 2. integer + fractional + optional quantity (e.g., "3.14" or "1.5 lakh")
        graph = self.optional_sign + (
            self.integer + self.quantity | self.integer + delete_space + self.fractional + self.optional_quantity
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

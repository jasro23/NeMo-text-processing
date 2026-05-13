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

from nemo_text_processing.text_normalization.kn.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst

# Map major currency to minor currency denomination in Kannada
major_minor_currencies = {
    "ರೂಪಾಯಿ": "ಪೈಸೆ",
    "ಡಾಲರ್": "ಸೆಂಟ್",
    "ಯೂರೋ": "ಸೆಂಟ್",
    "ಪೌಂಡ್": "ಪೆನ್ಸ್",
    "ಯೆನ್": "ಸೆನ್",
    "ವಾನ್": "ಜಿಯಾನ್",
    "ಟಾಕಾ": "ಪೈಸೆ",
    "ಲಿರಾ": "ಕುರುಶ್",
    "ನೈರಾ": "ಕೋಬೋ",
}


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಹನ್ನೆರಡು" } -> ಹನ್ನೆರಡು ರೂಪಾಯಿ
        money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐವತ್ತು" currency_min: "centiles" } -> ಹನ್ನೆರಡು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ
        money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಸೊನ್ನೆ" fractional_part: "ಐವತ್ತು" currency_min: "centiles" } -> ಐವತ್ತು ಪೈಸೆ

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        # Handles major denominations only
        # For each currency, create graph that reorders: currency_maj + integer_part -> integer currency
        major_only_graphs = []
        major_minor_graphs = []
        minor_graphs = []

        for major, minor in major_minor_currencies.items():
            # Delete currency_maj field and insert the currency name at the end
            delete_currency = (
                pynutil.delete('currency_maj: "')
                + pynutil.delete(major)
                + pynutil.delete('"')
                + pynutil.delete(NEMO_SPACE)
            )
            insert_currency = pynutil.insert(" " + major)
            
            # Major only: "50 rupees"
            # Input: currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು"
            # Output: ಐವತ್ತು ರೂಪಾಯಿ
            graph_major = delete_currency + integer_part + insert_currency
            major_only_graphs.append(graph_major)
            
            # Major + Minor: "50 rupees 50 paise"
            # Input: currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" fractional_part: "ಐವತ್ತು" currency_min: "centiles"
            # Output: ಐವತ್ತು ರೂಪಾಯಿ ಐವತ್ತು ಪೈಸೆ
            graph_minor = pynutil.delete('currency_min: "') + pynini.cross("centiles", minor) + pynutil.delete('"')
            graph_major_minor = (
                delete_currency
                + integer_part
                + insert_currency
                + pynutil.delete(NEMO_SPACE)
                + pynutil.insert(" ")
                + fractional_part
                + pynutil.delete(NEMO_SPACE)
                + pynutil.insert(" ")
                + graph_minor
            )
            major_minor_graphs.append(graph_major_minor)

            # Minor only: "50 paise" (when integer is zero)
            # Input: currency_maj: "ರೂಪಾಯಿ" integer_part: "ಸೊನ್ನೆ" fractional_part: "ಐವತ್ತು" currency_min: "centiles"
            # Output: ಐವತ್ತು ಪೈಸೆ
            graph_minor_only = (
                pynutil.delete('currency_maj: "')
                + pynutil.delete(major)
                + pynutil.delete('"')
                + pynutil.delete(NEMO_SPACE)
                + pynutil.delete('integer_part: "ಸೊನ್ನೆ"')
                + pynutil.delete(NEMO_SPACE)
                + fractional_part
                + pynutil.delete(NEMO_SPACE)
                + pynutil.insert(" ")
                + graph_minor
            )
            minor_graphs.append(graph_minor_only)

        graph_major_only = pynini.union(*major_only_graphs)
        graph_major_minor = pynini.union(*major_minor_graphs)
        graph_minor_only = pynini.union(*minor_graphs)

        graph = graph_major_only | graph_major_minor | pynutil.add_weight(graph_minor_only, -0.1)

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

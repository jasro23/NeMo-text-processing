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

from nemo_text_processing.text_normalization.kn.graph_utils import (
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path

unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, e.g.
        5kg -> measure { cardinal { integer: "ಐದು" } units: "ಕಿಲೋಗ್ರಾಂ" }
        10km -> measure { cardinal { integer: "ಹತ್ತು" } units: "ಕಿಲೋಮೀಟರ್" }
        3.5kg -> measure { decimal { integer_part: "ಮೂರು" fractional_part: "ಐದು" } units: "ಕಿಲೋಗ್ರಾಂ" }
        -5°C -> measure { cardinal { negative: "true" integer: "ಐದು" } units: "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        # Build cardinal graph from components (for faster compilation)
        cardinal_graph = (
            cardinal.zero
            | cardinal.digit
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
            | cardinal.graph_lakhs
            | cardinal.graph_ten_lakhs
        )

        # Decimal graph for measure
        point = pynutil.delete(".")
        decimal_integers = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        decimal_graph = decimal_integers + point + insert_space + decimal.graph_fractional

        # Unit wrapper
        unit = (
            pynutil.insert(" units: \"")
            + unit_graph
            + pynutil.insert("\"")
        )

        # Optional negative
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        # Cardinal + unit (e.g., 5kg)
        graph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        # Decimal + unit (e.g., 3.5kg)
        graph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        graph = graph_cardinal | pynutil.add_weight(graph_decimal, 0.1)

        self.graph = graph.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()

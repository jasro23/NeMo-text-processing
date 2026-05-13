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

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst, insert_space, delete_space
from nemo_text_processing.text_normalization.kn.utils import get_abs_path

# Quantity magnitudes for numbers like "1.5 lakh", "2.5 crore"
quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))

# Delete optional whitespace before quantity suffix
delete_optional_space = pynini.closure(pynutil.delete(" "), 0, 1)


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. ೧ ಲಕ್ಷ -> integer_part: "ಒಂದು" quantity: "ಲಕ್ಷ"
    e.g. ೧.೫ ಲಕ್ಷ -> integer_part: "ಒಂದು" fractional_part: "ಐದು" quantity: "ಲಕ್ಷ"
    e.g. 1.5ಲಕ್ಷ -> integer_part: "ಒಂದು" fractional_part: "ಐದು" quantity: "ಲಕ್ಷ" (no space)

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred

    # Cardinal + quantity (e.g., "1 lakh" or "1ಲಕ್ಷ")
    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_optional_space  # consume optional input space
        + insert_space  # insert output space
        + pynutil.insert("quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    # Decimal + quantity (e.g., "1.5 lakh" or "1.5ಲಕ್ಷ")
    res |= decimal + delete_optional_space + insert_space + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers, e.g.
        -೧೨.೫೦೦೬ -> decimal { negative: "true" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐದು ಸೊನ್ನೆ ಸೊನ್ನೆ ಆರು" }
        3.14 -> decimal { integer_part: "ಮೂರು" fractional_part: "ಒಂದು ನಾಲ್ಕು" }
        ೧.೫ ಲಕ್ಷ -> decimal { integer_part: "ಒಂದು" fractional_part: "ಐದು" quantity: "ಲಕ್ಷ" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        # Single digits for fractional part (read digit by digit)
        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.graph_without_leading_zeros

        # Fractional part: read each digit separately (e.g., ".14" -> "ಒಂದು ನಾಲ್ಕು")
        self.graph = graph_digit + pynini.closure(insert_space + graph_digit).optimize()

        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space,
            0,
            1,
        )

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

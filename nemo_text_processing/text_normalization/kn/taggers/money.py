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

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.kn.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        ₹೫೦ -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" }
        ₹50 -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" }
        ₹೫೦.೫೦ -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಐವತ್ತು" fractional_part: "ಐವತ್ತು" currency_min: "centiles" }
        ₹೦.೫೦ -> money { currency_maj: "ರೂಪಾಯಿ" integer_part: "ಸೊನ್ನೆ" fractional_part: "ಐವತ್ತು" currency_min: "centiles" }
    
    Note: 'centiles' is a placeholder handled by the verbalizer to apply the correct minor currency denomination

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.final_graph
        
        # For money fractions, we need to interpret .05 as 5 and .5 as 50
        # Build a special graph for 2-digit fractional amounts (00-99)
        # Single digit .X is interpreted as X0 (e.g., .5 = 50)
        digit = cardinal.digit
        zero = cardinal.zero
        teens_ties = cardinal.teens_and_ties
        
        # Two-digit fractions: 01-99 interpreted as cardinal number
        # .05 -> 5 (ಐದು), .50 -> 50 (ಐವತ್ತು), .99 -> 99 (ತೊಂಬತ್ತೊಂಬತ್ತು)
        # Note: .00 is excluded - ₹1.00 should normalize as just "ಒಂದು ರೂಪಾಯಿ"
        fraction_two_digit = (
            (pynutil.delete(pynini.union("0", "೦")) + digit)  # 01-09 -> 1-9
            | teens_ties  # 10-99
        )
        
        # Single-digit fraction: .X interpreted as X0 (e.g., .5 = 50, .1 = 10)
        single_digit_map = pynini.string_map([
            ("1", "ಹತ್ತು"), ("೧", "ಹತ್ತು"),
            ("2", "ಇಪ್ಪತ್ತು"), ("೨", "ಇಪ್ಪತ್ತು"),
            ("3", "ಮೂವತ್ತು"), ("೩", "ಮೂವತ್ತು"),
            ("4", "ನಲವತ್ತು"), ("೪", "ನಲವತ್ತು"),
            ("5", "ಐವತ್ತು"), ("೫", "ಐವತ್ತು"),
            ("6", "ಅರವತ್ತು"), ("೬", "ಅರವತ್ತು"),
            ("7", "ಎಪ್ಪತ್ತು"), ("೭", "ಎಪ್ಪತ್ತು"),
            ("8", "ಎಂಬತ್ತು"), ("೮", "ಎಂಬತ್ತು"),
            ("9", "ತೊಂಬತ್ತು"), ("೯", "ತೊಂಬತ್ತು"),
        ])
        
        currency_major = pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        
        # Fractional part for money: handle both 2-digit and 1-digit as cardinal
        fraction_graph = fraction_two_digit | single_digit_map
        fraction = pynutil.insert('fractional_part: "') + fraction_graph + pynutil.insert('"')
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')
        
        # For 3+ digit fractions, read digit-by-digit
        # Build: digit + space + digit + space + ... + digit (no trailing space)
        single_digit_only = digit | zero
        single_digit_with_space = single_digit_only + pynutil.insert(" ")
        # At least 3 digits: (d + space)+ + (d + space) + d = 3+ digits with spaces between
        digit_by_digit_graph = (
            pynini.closure(single_digit_with_space, 2) + single_digit_only  # 3+ digits, no trailing space
        )
        fraction_digit_by_digit = pynutil.insert('fractional_part: "') + digit_by_digit_graph + pynutil.insert('"')

        # No negative support for money (unusual use case)
        graph_major_only = currency_major + insert_space + integer
        
        # Match .00 and discard it (₹1.00 -> ₹1)
        delete_zero_fraction = pynutil.delete(pynini.union(".00", ".೦೦", ".0೦", ".೦0"))
        graph_major_with_zero_fraction = currency_major + insert_space + integer + delete_zero_fraction
        
        # Standard 1-2 digit fractions (as cardinal)
        graph_major_and_minor = (
            currency_major
            + insert_space
            + integer
            + pynini.cross(".", " ")
            + fraction
            + insert_space
            + currency_minor
        )
        
        # 3+ digit fractions (digit-by-digit)
        graph_major_and_minor_digit_by_digit = (
            currency_major
            + insert_space
            + integer
            + pynini.cross(".", " ")
            + fraction_digit_by_digit
            + insert_space
            + currency_minor
        )

        graph_currencies = graph_major_only | graph_major_with_zero_fraction | graph_major_and_minor | graph_major_and_minor_digit_by_digit

        graph = graph_currencies.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph

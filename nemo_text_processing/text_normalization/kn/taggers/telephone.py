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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_KN_DIGIT,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path

# Valid starting digits for Indian mobile numbers (6-9)
KN_MOBILE_START_DIGITS = pynini.union("೬", "೭", "೮", "೯", "6", "7", "8", "9").optimize()
# Valid starting digits for Indian landline STD codes (2-6)
KN_LANDLINE_START_DIGITS = pynini.union("೨", "೩", "೪", "೬", "2", "3", "4", "6").optimize()

KN_ZERO_DIGIT = pynini.union("0", "೦")
delete_zero = pynutil.delete(KN_ZERO_DIGIT)
delete_zero_optional = pynini.closure(delete_zero, 0, 1)
insert_sonne = pynutil.insert("ಸೊನ್ನೆ") + insert_space

# Load the number mappings from TSV files
digit_to_word = pynini.string_file(get_abs_path("data/telephone/number.tsv"))
digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
mobile_context = pynini.string_file(get_abs_path("data/telephone/mobile_context.tsv"))
landline_context = pynini.string_file(get_abs_path("data/telephone/landline_context.tsv"))
credit_context = pynini.string_file(get_abs_path("data/telephone/credit_context.tsv"))
pincode_context = pynini.string_file(get_abs_path("data/telephone/pincode_context.tsv"))

# Reusable optimized graph for any digit token
num_token = pynini.union(digit_to_word, digits, zero).optimize()


def get_context(keywords: pynini.Fst):
    """
    Creates context graphs that look for keywords before or after a number.
    Keywords and window words are preserved in output (matching Hindi behavior).
    Returns (context_before, context_after) FSTs.
    """
    all_digits = pynini.union(NEMO_KN_DIGIT, NEMO_DIGIT)
    
    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + pynini.accep(NEMO_SPACE)
    
    window = pynini.closure(word, 0, 5)
    
    # Accept keywords and window (preserve in output, matching Hindi)
    before = pynini.closure(keywords + pynini.accep(NEMO_SPACE) + window, 0, 1)
    after = pynini.closure(pynutil.delete(NEMO_SPACE) + window + keywords, 0, 1)
    
    return before.optimize(), after.optimize()


def generate_mobile(context_keywords: pynini.Fst) -> pynini.Fst:
    """Generate mobile number graph with context."""
    context_before, context_after = get_context(context_keywords)
    
    # Mobile numbers must start with 6-9
    mobile_start_digit = pynini.union(
        KN_MOBILE_START_DIGITS @ digits,
        KN_MOBILE_START_DIGITS @ digit_to_word
    )
    
    # Country code: + followed by 1-3 digits, with optional hyphen after
    country_code_digits = pynini.closure(num_token + insert_space, 1, 3)
    # Optional hyphen/space after country code (e.g., +91-9876... or +91 9876...)
    optional_separator = pynini.closure(pynini.union(pynutil.delete("-"), delete_space), 0, 1)
    country_code = (
        pynutil.insert("country_code: \"")
        + context_before
        + pynini.cross("+", "ಪ್ಲಸ್")
        + insert_space
        + country_code_digits
        + pynutil.insert("\" ")
        + optional_separator
    )
    
    # Optional extension (1-3 digits after main number)
    extension_optional = pynini.closure(
        pynutil.insert("extension: \"")
        + pynini.closure(num_token + insert_space, 1, 3)
        + context_after
        + pynutil.insert("\" ")
        + delete_space,
        0,
        1,
    )
    
    # 10-digit mobile number (first digit 6-9, then exactly 9 more digits)
    number_part = mobile_start_digit + insert_space + pynini.closure(num_token + insert_space, 9, 9)
    
    # Mobile without country code - two paths:
    # 1. With leading 0: delete 0 and insert ಸೊನ್ನೆ
    # 2. Without leading 0: no zero in output
    number_with_leading_zero = (
        pynutil.insert("number_part: \"")
        + context_before
        + pynutil.delete(KN_ZERO_DIGIT)
        + insert_sonne
        + number_part
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )
    
    number_without_leading_zero = (
        pynutil.insert("number_part: \"")
        + context_before
        + number_part
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )
    
    number_without_country = pynini.union(number_with_leading_zero, number_without_leading_zero)
    
    # Mobile with country code
    number_with_country = (
        country_code
        + pynutil.insert("number_part: \"")
        + number_part
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    )
    
    return (pynini.union(number_with_country, number_without_country) + extension_optional).optimize()


def get_landline(std_length: int, context_keywords: pynini.Fst) -> pynini.Fst:
    """Generate landline graph for a specific STD code length.
    
    With context keyword present, any digit is accepted for local number start.
    The 2/3/4/6 restriction was for disambiguation without context.
    """
    context_before, context_after = get_context(context_keywords)
    
    # STD code part - two paths for optional leading 0
    std_code_with_zero = (
        pynutil.delete(KN_ZERO_DIGIT) + insert_sonne + pynini.closure(num_token + insert_space, std_length, std_length)
    )
    std_code_without_zero = pynini.closure(num_token + insert_space, std_length, std_length)
    std_code_graph = pynini.union(std_code_with_zero, std_code_without_zero)
    
    # Local number part (total digits = 10 - STD length)
    # Accept any digit for local number start when context is present
    landline_digit_count = 10 - std_length
    landline_graph = pynini.closure(num_token + insert_space, landline_digit_count, landline_digit_count)
    
    # Optional separator between STD and local number
    separator_optional = pynini.closure(pynini.union(pynini.cross("-", ""), pynini.cross(".", "")), 0, 1)
    
    # STD code in brackets: (080) or ( 080 ) or 0(80) etc.
    # Handle optional leading 0 before brackets
    leading_zero_before_bracket = pynini.closure(pynutil.delete(KN_ZERO_DIGIT) + insert_sonne, 0, 1)
    std_code_in_brackets = (
        leading_zero_before_bracket
        + delete_space
        + pynutil.delete("(")
        + pynini.closure(delete_space, 0, 1)
        + std_code_graph
        + pynini.closure(delete_space, 0, 1)
        + pynutil.delete(")")
    )
    
    std_part = pynini.union(std_code_graph, std_code_in_brackets)
    
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + std_part
        + separator_optional
        + delete_space
        + landline_graph
        + context_after
        + pynutil.insert("\" ")
    ).optimize()


def generate_landline(context_keywords: pynini.Fst) -> pynini.Fst:
    """Generate landline graph for all STD code lengths (2-7 digits)."""
    graph = (
        get_landline(2, context_keywords)
        | get_landline(3, context_keywords)
        | get_landline(4, context_keywords)
        | get_landline(5, context_keywords)
        | get_landline(6, context_keywords)
        | get_landline(7, context_keywords)
    )
    return graph.optimize()


def generate_credit(context_keywords: pynini.Fst) -> pynini.Fst:
    """Generate credit card number graph (16 digits, typically in 4 groups of 4)."""
    context_before, context_after = get_context(context_keywords)
    
    # 4 digits per group, 4 groups
    digit_group = pynini.closure(num_token + insert_space, 4, 4)
    separator_optional = pynini.closure(pynini.union(pynini.cross("-", ""), pynini.cross(" ", "")), 0, 1)
    
    # 16 digits total (with optional separators)
    credit_graph = (
        digit_group
        + pynini.closure(separator_optional + digit_group, 3, 3)
    )
    
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + credit_graph
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


def generate_pincode(context_keywords: pynini.Fst) -> pynini.Fst:
    """Generate pincode graph (exactly 6 digits)."""
    context_before, context_after = get_context(context_keywords)
    return (
        pynutil.insert("number_part: \"")
        + context_before
        + pynini.closure(num_token + insert_space, 6, 6)
        + context_after
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


def generate_tollfree() -> pynini.Fst:
    """Generate toll-free number graph (1800-xxx-xxx or 18001234567).
    
    Indian toll-free numbers start with 1800, followed by 6-7 more digits.
    Total: 10-11 digits (1800 + 6-7)
    
    Examples:
        1800-123-456 -> ಒಂದು ಎಂಟು ಸೊನ್ನೆ ಸೊನ್ನೆ ಒಂದು ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು
        18001234567 -> ಒಂದು ಎಂಟು ಸೊನ್ನೆ ಸೊನ್ನೆ ...
    """
    # 1800 prefix - using digit mapping like other phone functions
    digit_1 = pynini.cross("1", "") @ pynini.union(digit_to_word, digits)
    digit_8 = pynini.cross("8", "") @ pynini.union(digit_to_word, digits)
    digit_0 = pynini.cross("0", "") @ pynini.union(digit_to_word, digits, zero)
    
    prefix_1800 = (
        pynini.cross("1", "ಒಂದು") + insert_space
        + pynini.cross("8", "ಎಂಟು") + insert_space
        + pynini.cross("0", "ಸೊನ್ನೆ") + insert_space
        + pynini.cross("0", "ಸೊನ್ನೆ") + insert_space
    )
    
    # Optional separator (hyphen or space)
    separator_optional = pynini.closure(
        pynini.union(pynutil.delete("-"), pynutil.delete(" ")),
        0, 1
    )
    
    # Remaining digits with separators
    # Format: 1800-xxx-xxx (groups of 3) or 1800-xxx-xxxx (3+4)
    group_of_3 = pynini.closure(num_token + insert_space, 3, 3)
    group_of_4 = pynini.closure(num_token + insert_space, 4, 4)
    
    # 1800-123-456 (10 total: 1800 + 3 + 3)
    remaining_3_3 = group_of_3 + separator_optional + group_of_3
    # 1800-123-4567 (11 total: 1800 + 3 + 4)
    remaining_3_4 = group_of_3 + separator_optional + group_of_4
    # 1800-1234-567 (11 total: 1800 + 4 + 3)
    remaining_4_3 = group_of_4 + separator_optional + group_of_3
    
    remaining_with_separators = pynini.union(remaining_3_3, remaining_3_4, remaining_4_3)
    
    # Continuous format: 1800xxxxxx (6 digits) or 1800xxxxxxx (7 digits)
    remaining_continuous = pynini.closure(num_token + insert_space, 6, 7)
    
    remaining_part = pynini.union(remaining_with_separators, remaining_continuous)
    
    return (
        pynutil.insert("number_part: \"")
        + prefix_1800
        + separator_optional
        + remaining_part
        + pynutil.insert("\" ")
        + delete_space
    ).optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, credit cards, pincodes, and toll-free numbers.
    Uses context keywords to disambiguate from cardinal numbers.
    
    Examples:
        ಮೊಬೈಲ್ 9876543210 -> telephone { number_part: "ಒಂಬತ್ತು ಎಂಟು ಏಳು ..." }
        +91 9876543210 -> telephone { country_code: "ಪ್ಲಸ್ ಒಂಬತ್ತು ಒಂದು " number_part: "..." }
        +91-9876543210 -> telephone { country_code: "ಪ್ಲಸ್ ಒಂಬತ್ತು ಒಂದು " number_part: "..." }
        ಪಿನ್‌ಕೋಡ್ 560001 -> telephone { number_part: "ಐದು ಆರು ಸೊನ್ನೆ ಸೊನ್ನೆ ಸೊನ್ನೆ ಒಂದು " }
        1800-123-456 -> telephone { number_part: "ಒಂದು ಎಂಟು ಸೊನ್ನೆ ಸೊನ್ನೆ ..." }
        
    Context keywords trigger telephone classification:
        - Mobile: ನಂಬರ್, ಮೊಬೈಲ್, ಫೋನ್, ದೂರವಾಣಿ, ಕಾಲ್, ಕರೆ
        - Credit: ಕಾರ್ಡ್, ಕ್ರೆಡಿಟ್
        - Pincode: ಪಿನ್, ಕೋಡ್, ಪಿನ್‌ಕೋಡ್
        
    Toll-free numbers (1800-xxx-xxx) are recognized without context keywords.

    Args:
        cardinal: CardinalFst (not used, kept for API consistency)
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        mobile_number = generate_mobile(mobile_context)
        landline = generate_landline(landline_context)
        credit_card = generate_credit(credit_context)
        pincode = generate_pincode(pincode_context)
        tollfree = generate_tollfree()

        graph = (
            pynutil.add_weight(tollfree, 0.6)  # Highest priority for 1800 numbers
            | pynutil.add_weight(mobile_number, 0.7)
            | pynutil.add_weight(landline, 0.8)
            | pynutil.add_weight(credit_card, 0.9)
            | pynutil.add_weight(pincode, 1.0)
        )

        self.fst = self.add_tokens(graph.optimize())

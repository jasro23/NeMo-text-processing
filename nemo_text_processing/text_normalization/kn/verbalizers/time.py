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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" } -> ಹತ್ತು ಗಂಟೆ ಮೂವತ್ತು ನಿಮಿಷ
        time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" seconds: "ನಲವತ್ತೈದು" } -> ಹತ್ತು ಗಂಟೆ ಮೂವತ್ತು ನಿಮಿಷ ನಲವತ್ತೈದು ಸೆಕೆಂಡು
        time { hours: "ಹತ್ತು" } -> ಹತ್ತು ಗಂಟೆ
        time { hours: "ಹತ್ತು" suffix: "ಬೆಳಿಗ್ಗೆ" } -> ಹತ್ತು ಗಂಟೆ ಬೆಳಿಗ್ಗೆ

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour = (
            pynutil.delete("hours: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + insert_space
        )

        minute = (
            pynutil.delete("minutes: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + insert_space
        )

        second = (
            pynutil.delete("seconds: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + insert_space
        )

        # AM/PM suffix (ಬೆಳಿಗ್ಗೆ = morning, ಸಂಜೆ = evening)
        suffix = (
            pynutil.delete("suffix: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_suffix = pynini.closure(delete_space + insert_space + suffix, 0, 1)

        # Kannada time markers
        insert_gante = pynutil.insert("ಗಂಟೆ")  # hour marker
        insert_nimisha = pynutil.insert("ನಿಮಿಷ")  # minute marker
        insert_sekendu = pynutil.insert("ಸೆಕೆಂಡು")  # second marker

        # hour:minute:second (no AM/PM with seconds)
        graph_hms = (
            hour
            + delete_space
            + insert_gante
            + insert_space
            + minute
            + delete_space
            + insert_nimisha
            + insert_space
            + second
            + delete_space
            + insert_sekendu
        )

        # hour:minute with optional AM/PM
        graph_hm = (
            hour
            + delete_space
            + insert_gante
            + insert_space
            + minute
            + delete_space
            + insert_nimisha
            + optional_suffix
        )

        # hour only with optional AM/PM
        graph_h = hour + delete_space + insert_gante + optional_suffix

        self.graph = graph_hms | graph_hm | graph_h

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()

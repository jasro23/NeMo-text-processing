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

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.kn.utils import get_abs_path

# Time patterns - support both Kannada and Arabic digits
KN_DOUBLE_ZERO = pynini.union("೦೦", "00")

# AM/PM patterns (case-insensitive)
AM_PATTERN = pynini.union("AM", "am", "Am", "aM", "ಎ ಎಂ", "ಎಎಂ")
PM_PATTERN = pynini.union("PM", "pm", "Pm", "pM", "ಪಿ ಎಂ", "ಪಿಎಂ")

hours_graph = pynini.string_file(get_abs_path("data/time/hours.tsv"))
minutes_graph = pynini.string_file(get_abs_path("data/time/minutes.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        10:30 -> time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" }
        10:30:45 -> time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" seconds: "ನಲವತ್ತೈದು" }
        10:00 -> time { hours: "ಹತ್ತು" }
        10:30 AM -> time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" suffix: "ಬೆಳಿಗ್ಗೆ" }
        10:30 PM -> time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" suffix: "ಸಂಜೆ" }
        ೧೦:೩೦ -> time { hours: "ಹತ್ತು" minutes: "ಮೂವತ್ತು" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        delete_colon = pynutil.delete(":")

        self.hours = pynutil.insert("hours: \"") + hours_graph + pynutil.insert("\"")
        self.minutes = pynutil.insert(" minutes: \"") + minutes_graph + pynutil.insert("\"")
        self.seconds = pynutil.insert(" seconds: \"") + minutes_graph + pynutil.insert("\"")

        # AM/PM suffix handling - use loanwords ಎ ಎಂ / ಪಿ ಎಂ
        optional_space = pynini.closure(delete_space, 0, 1)
        am_suffix = optional_space + pynini.cross(AM_PATTERN, "") + pynutil.insert(" suffix: \"ಎ ಎಂ\"")
        pm_suffix = optional_space + pynini.cross(PM_PATTERN, "") + pynutil.insert(" suffix: \"ಪಿ ಎಂ\"")
        optional_suffix = pynini.closure(am_suffix | pm_suffix, 0, 1)

        # hour:minute:second (no AM/PM with seconds)
        graph_hms = (
            self.hours + delete_colon + self.minutes + delete_colon + self.seconds
        )

        # hour:minute with optional AM/PM
        graph_hm = self.hours + delete_colon + self.minutes + optional_suffix

        # hour:00 (just the hour) with optional AM/PM - HIGHEST PRIORITY
        graph_h = self.hours + delete_colon + pynutil.delete(KN_DOUBLE_ZERO) + optional_suffix

        # hour AM/PM (no colon, e.g., "10 AM")
        graph_h_ampm = self.hours + (am_suffix | pm_suffix)

        final_graph = (
            pynutil.add_weight(graph_h, 0.05)  # X:00 gets highest priority
            | pynutil.add_weight(graph_hms, 0.1)
            | pynutil.add_weight(graph_hm, 0.2)
            | pynutil.add_weight(graph_h_ampm, 0.3)
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

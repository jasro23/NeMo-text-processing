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

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst
from nemo_text_processing.text_normalization.kn.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.kn.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.kn.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.kn.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.kn.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.kn.verbalizers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.kn.verbalizers.time import TimeFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars for Kannada.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)

        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst

        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst

        money = MoneyFst()
        money_graph = money.fst

        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst

        telephone = TelephoneFst(deterministic=deterministic)
        telephone_graph = telephone.fst

        time = TimeFst()
        time_graph = time.fst

        measure = MeasureFst(cardinal=cardinal, decimal=decimal)
        measure_graph = measure.fst

        graph = (
            cardinal_graph
            | decimal_graph
            | money_graph
            | ordinal_graph
            | telephone_graph
            | time_graph
            | measure_graph
        )

        self.fst = graph

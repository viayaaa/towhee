# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import torch

from towhee.models.allinone.allinone import VCOPHeader


class AllinoneTest(unittest.TestCase):
    def test_allinone(self):
        vcoph = VCOPHeader(feature_size=4)
        x = torch.rand(2, 3, 4)
        out = vcoph(x)
        self.assertTrue(out.shape == torch.Size([2, 6]))


if __name__ == "__main__":
    unittest.main()

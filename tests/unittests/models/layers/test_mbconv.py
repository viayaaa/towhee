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
from towhee.models.layers.mbconv import MBConv


class MbConvTest(unittest.TestCase):

    def test_mb_conv(self):
        model = MBConv(in_channels=30, out_channels=30)
        x = torch.randn(1, 30, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (1, 30, 28, 28))


if __name__ == '__main__':
    unittest.main()

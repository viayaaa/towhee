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
from towhee.models.uniformer.uniformer import Uniformer

class TestUniFormer(unittest.TestCase):
    def test_uniformer(self):
        model=Uniformer()
        x=torch.randn(1,3,3,32,32)
        out=model(x)
        self.assertTrue(out.shape == (1, 400))

if __name__ == "__main__":
    unittest.main()

/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class RBMLayer extends WeightLayer {

   public RBMLayer(int inputSize, int outputSize) {
      super(inputSize, outputSize, Activation.SIGMOID,
            NDArrayInitializer.glorotAndBengioSigmoid, 0, 0);
   }

   @Override
   public Layer copy() {
      return new RBMLayer(getInputSize(), getOutputSize());
   }

   @Override
   public void preTrain(Dataset dataset) {
      NDArray weights = getWeights();

      // Do something to the weights
   }

}//END OF RBMLayer

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

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.optimization.CostFunction;
import com.gengoai.apollo.optimization.CostGradientTuple;
import com.gengoai.apollo.optimization.GradientParameter;
import com.gengoai.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class FeedForwardCostFunction implements CostFunction<FeedForwardNetwork> {
   LossFunction lossFunction;

   public FeedForwardCostFunction(LossFunction lossFunction) {
      this.lossFunction = lossFunction;
   }

   @Override
   public CostGradientTuple evaluate(NDArray input, FeedForwardNetwork network) {
      NDArray[] ai = new NDArray[network.layers.size()];
      NDArray cai = input;
      NDArray Y = input.getLabelAsNDArray();
      for (int i = 0; i < network.layers.size(); i++) {
         cai = network.layers.get(i).forward(cai);
         ai[i] = cai;
      }
      if (cai.numRows() == 1) { //If Binary, only take the first row of the Y
         Y = Y.getVector(1, Axis.ROW);
      }
      double loss = lossFunction.loss(cai, Y) / input.numCols();
      NDArray dz = lossFunction.derivative(cai, Y);
      return CostGradientTuple.of(loss,
                                  GradientParameter.of(dz, dz),
                                  ai);
   }
}// END OF FeedForwardCostFunction

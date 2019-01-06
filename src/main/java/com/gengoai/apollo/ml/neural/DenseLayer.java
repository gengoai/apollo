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


import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.activation.TanH;

/**
 * @author David B. Bracewell
 */
public class DenseLayer extends WeightLayer {
   public DenseLayer(int inputSize, int outputSize, Activation activation, NDArrayInitializer NDArrayInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, NDArrayInitializer, l1, l2);
   }

   public DenseLayer(WeightLayer layer) {
      super(layer);
   }

   public static Builder relu() {
      return new Builder().activation(Activation.RELU);
   }

   public static Builder sigmoid() {
      return new Builder().activation(Activation.SIGMOID);
   }

   public static Builder linear() {
      return new Builder().activation(Activation.LINEAR);
   }

   public static Builder tanH() {
      return new Builder().activation(new TanH());
   }

   @Override
   public Layer copy() {
      return new DenseLayer(this);
   }

   public static class Builder extends WeightLayerBuilder<Builder> {

      @Override
      public Layer build() {
         return new DenseLayer(getInputSize(), getOutputSize(), getActivation(), this.getInitializer(), getL1(),
                               getL2());
      }
   }

}// END OF DenseLayer

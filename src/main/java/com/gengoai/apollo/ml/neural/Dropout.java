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
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.tuple.Tuple2;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class Dropout extends Layer {
   double rate;

   protected Dropout(int inputSize, int outputSize, double rate) {
      super(inputSize, outputSize);
      this.rate = rate;
   }

   public static DropoutBuilder builder() {
      return new DropoutBuilder();
   }

   public static DropoutBuilder builder(double rate) {
      return new DropoutBuilder().rate(rate);
   }

   @Override
   public NDArray backward(NDArray input, NDArray output, NDArray delta, double learningRate, int layerIndex, int iteration) {
      return delta;
   }

   @Override
   public BackpropResult backward(NDArray input, NDArray output, NDArray delta, boolean calculateDelta) {
      return BackpropResult.from(delta, NDArrayFactory.DEFAULT().empty(),
                                 NDArrayFactory.DEFAULT().empty());
   }

   @Override
   public Tuple2<NDArray, Double> backward(WeightUpdate updater, NDArray input, NDArray output, NDArray delta, int iteration, boolean calcuateDelta) {
      return $(delta, 0d);
   }

   @Override
   public Layer copy() {
      return new Dropout(getInputSize(), getOutputSize(), rate);
   }

   @Override
   NDArray forward(NDArray input) {
      NDArray mask = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand, input.numRows(), input.numCols())
                               .test(x -> x < rate);
      return input.mul(mask).divi((float) rate);
   }

   @Override
   public NDArray getBias() {
      return NDArrayFactory.DEFAULT().empty();
   }

   @Override
   public NDArray getWeights() {
      return NDArrayFactory.DEFAULT().empty();
   }

   @Override
   public boolean trainOnly() {
      return true;
   }

   @Override
   public double update(WeightUpdate weightUpdate, NDArray wGrad, NDArray bBrad, int iteration) {
      return 0;
   }

   @Override
   public void update(NDArray[] weights, NDArray[] bias) {

   }

   public static class DropoutBuilder extends LayerBuilder<DropoutBuilder> {
      public double getRate() {
         return rate;
      }

      double rate = 0.5;

      @Override
      public Layer build() {
         return new Dropout(getInputSize(), getOutputSize(), getRate());
      }

      public DropoutBuilder rate(double rate) {
         this.rate = rate;
         return this;
      }
   }
}// END OF Dropout

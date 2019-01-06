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
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.optimization.GradientParameter;
import com.gengoai.apollo.optimization.LinearModelParameters;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.conversion.Cast;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public abstract class WeightLayer extends Layer implements LinearModelParameters {
   private static final long serialVersionUID = 1L;
   protected final Activation activation;
   protected final double l1;
   protected final double l2;
   protected NDArray bias;
   protected transient NDArray v;
   protected NDArray weights;

   public WeightLayer(int inputSize, int outputSize, Activation activation, NDArrayInitializer NDArrayInitializer, double l1, double l2) {
      super(inputSize, outputSize);
      this.activation = activation;
      this.weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer, outputSize, inputSize);
      this.bias = NDArrayFactory.DEFAULT().zeros(outputSize);
      this.v = NDArrayFactory.DEFAULT().zeros(outputSize, inputSize);
      this.l1 = l1;
      this.l2 = l2;
   }

   public WeightLayer(WeightLayer layer) {
      super(layer.getInputSize(), layer.getOutputSize());
      this.activation = layer.getActivation();
      this.bias = layer.bias.copy();
      this.weights = layer.weights.copy();
      this.l1 = layer.l1;
      this.l2 = layer.l2;
      this.v = NDArrayFactory.DEFAULT().zeros(layer.getOutputSize(), layer.getInputSize());
   }

   @Override
   public BackpropResult backward(NDArray input, NDArray output, NDArray delta, boolean calculateDelta) {
      delta.muli(activation.valueGradient(output));
      NDArray dzOut = calculateDelta
                      ? weights.T().mmul(delta)
                      : null;
      NDArray dw = delta.mmul(input.T());
      NDArray db = delta.sum(Axis.ROW);
      return BackpropResult.from(dzOut, dw, db);
   }

   @Override
   public Tuple2<NDArray, Double> backward(WeightUpdate updater, NDArray input, NDArray output, NDArray delta, int iteration, boolean calcuateDelta) {
      return updater.update(this, input, output, delta.muli(activation.valueGradient(output)), iteration,
                            calcuateDelta);
   }

   @Override
   public NDArray backward(NDArray input, NDArray output, NDArray delta, double learningRate, int layerIndex, int iteration) {
      delta.muli(activation.valueGradient(output));
      NDArray dzOut = layerIndex > 0
                      ? weights.T().mmul(delta)
                      : null;
      NDArray dw = delta.mmul(input.T())
                        .divi(input.numCols());
      NDArray db = delta.sum(Axis.ROW)
                        .divi(input.numCols());
      v.muli(0.9f).subi(dw.muli((float) learningRate));
      weights.addi(v);
      bias.subi(db.muli((float) learningRate));
      l1Update(learningRate, iteration);
      return dzOut;
   }

   @Override
   public NDArray forward(NDArray input) {
      return activation.apply(weights.mmul(input).addi(bias, Axis.COLUMN));
   }

   @Override
   public Activation getActivation() {
      return activation;
   }

   @Override
   public NDArray getBias() {
      return bias;
   }

   @Override
   public NDArray getWeights() {
      return weights;
   }

   protected void l1Update(double learningRate, int iteration) {
      if (l1 > 0) {
         //L1 Regularization
         double shrinkage = l1 * (learningRate / iteration);
         weights.mapi(x -> {
            double xp = FastMath.signum(x) * FastMath.max(0, FastMath.abs(x) - shrinkage);
            if (FastMath.abs(xp) < 1e-9) {
               return 0d;
            }
            return xp;
         });
      }
   }

   @Override
   public int numberOfFeatures() {
      return getInputSize();
   }

   @Override
   public int numberOfLabels() {
      return getOutputSize();
   }

   @Override
   public double update(WeightUpdate weightUpdate, NDArray wGrad, NDArray bBrad, int iteration) {
      return weightUpdate.update(this, GradientParameter.of(wGrad, bBrad), iteration);
   }

   @Override
   public void update(NDArray[] weights, NDArray[] bias) {
      NDArray wP = this.weights.getFactory().zeros(getOutputSize(), getInputSize());
      NDArray bP = this.weights.getFactory().zeros(getOutputSize());
      for (int i = 0; i < weights.length; i++) {
         wP.addi(weights[i]);
         bP.addi(bias[i]);
      }
      if (weights.length > 0) {
         wP.divi(weights.length);
         bP.divi(weights.length);
         this.weights = wP;
         this.bias = bP;
      }
   }

   protected static abstract class WeightLayerBuilder<T extends WeightLayerBuilder> extends LayerBuilder<T> {
      private Activation activation = Activation.SIGMOID;
      private NDArrayInitializer initializer = NDArrayInitializer.glorotAndBengioSigmoid;
      private double l1 = 0;

      public Activation getActivation() {
         return activation;
      }

      public NDArrayInitializer getInitializer() {
         return initializer;
      }

      public double getL1() {
         return l1;
      }

      public double getL2() {
         return l2;
      }

      private double l2 = 0;

      public T activation(Activation activation) {
         this.activation = activation;
         return Cast.as(this);
      }

      public T l1(double l1) {
         this.l1 = l1;
         return Cast.as(this);
      }

      public T l2(double l2) {
         this.l2 = l2;
         return Cast.as(this);
      }

      public T weightInitializer(NDArrayInitializer initializer) {
         this.initializer = initializer;
         return Cast.as(this);
      }

   }

}// END OF WeightLayer

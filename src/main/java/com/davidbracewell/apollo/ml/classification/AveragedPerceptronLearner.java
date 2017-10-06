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
 */

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.logging.Logger;
import lombok.Getter;
import lombok.Setter;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * <p>Trains a binary Averaged Perceptron model.</p>
 *
 * @author David B. Bracewell
 */
public class AveragedPerceptronLearner extends BinaryClassifierLearner {
   private static final long serialVersionUID = 1L;
   private static Logger log = Logger.getLogger(AveragedPerceptronLearner.class);
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private double learningRate = 1.0;
   private NDArray totalWeights;
   private NDArray stamps;
   private double totalBias;
   private double biasStamps;
   @Getter
   @Setter
   private double tolerance = 1e-9;
   @Getter
   @Setter
   private boolean verbose = false;

   /**
    * Instantiates a new Averaged perceptron learner.
    */
   public AveragedPerceptronLearner() {

   }

   /**
    * Instantiates a new Averaged perceptron learner.
    *
    * @param maxIterations the maximum number of iterations (default 100)
    * @param learningRate  the learning rate to control how fast weights are changed (default 1.0)
    * @param tolerance     the error tolerance used to determine if the algorithm has converged (default 0.0001)
    */
   public AveragedPerceptronLearner(int maxIterations, double learningRate, double tolerance) {
      this.maxIterations = maxIterations;
      this.learningRate = learningRate;
      this.tolerance = tolerance;
   }

   private double convertY(double real, double trueLabel) {
      return (real == trueLabel) ? 1.0 : 0.0;
   }

   @Override
   protected void resetLearnerParameters() {
      this.totalBias = 0;
      this.biasStamps = 0;
      this.stamps = null;
      this.totalWeights = null;
   }


   @Override
   protected Classifier trainForLabel(Dataset<Instance> dataset, double trueLabel) {
      BinaryGLM model = new BinaryGLM(this);

      totalWeights = NDArrayFactory.defaultFactory().zeros(model.numberOfFeatures());
      stamps = NDArrayFactory.defaultFactory().zeros(model.numberOfFeatures());
      model.weights = NDArrayFactory.defaultFactory().zeros(model.numberOfFeatures());

      double c = 1d;
      double oldError = 0;
      double oldOldError = 0;
      for (int iteration = 0; iteration < maxIterations; iteration++) {
         double error = 0;
         double count = 0;
         for (Instance instance : dataset) {
            NDArray v = instance.toVector(getEncoderPair());
            count++;
            double y = convertY(v.getLabel(), trueLabel);
            double yHat = model.classify(v).getEncodedResult();

            if (y != yHat) {
               error++;
               double eta = learningRate * (y - yHat);
               for (NDArray.Entry entry : Collect.asIterable(v.sparseIterator())) {
                  updateFeature(model, entry.getIndex(), c, eta);
               }
               double timeSpan = c - biasStamps;
               totalBias += (timeSpan * model.bias);
               model.bias += eta;
               biasStamps = c;
            }
            c++;
         }

         if (verbose) {
            log.info("iteration={0} errorRate={1,number,0.00%} (true={2})", iteration, (error / count),
                     model.getLabelEncoder().decode(trueLabel));
         }
         if (error == 0) {
            break;
         }
         error /= count;
         if (iteration > 2) {
            if (error != count && Math.abs(error - oldError) < tolerance && Math.abs(error - oldOldError) < tolerance) {
               break;
            }
         }
         oldOldError = oldError;
         oldError = error;
      }

      double time = c;
      for (NDArray.Entry entry : Collect.asIterable(totalWeights.sparseIterator())) {
         double total = totalWeights.get(entry.getIndex());
         total += (time - stamps.get(entry.getIndex())) * model.weights.get(entry.getIndex());
         total = new BigDecimal(total / time).setScale(3, RoundingMode.HALF_UP).doubleValue();
         model.weights.set(entry.getIndex(), total);
      }
      double total = totalBias;
      total += (time - biasStamps) * model.bias;
      total = new BigDecimal(total / time).setScale(3, RoundingMode.HALF_UP).doubleValue();
      model.bias = total;
      model.activation = Activation.LINEAR;
      return model;
   }

   private void updateFeature(BinaryGLM model, int featureId, double time, double value) {
      double timeAtWeight = time - stamps.get(featureId);
      double curWeight = model.weights.get(featureId);
      totalWeights.increment(featureId, timeAtWeight * curWeight);
      stamps.set(featureId, time);
      model.weights.set(featureId, curWeight + value);
   }

}//END OF AveragedPerceptronLearner

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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Collect;
import com.gengoai.guava.common.base.Stopwatch;
import com.gengoai.guava.common.collect.Lists;
import com.gengoai.logging.Logger;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;

import java.text.DecimalFormat;
import java.util.List;

/**
 * <p>Trains {@link StructuredPerceptron}, or Collin's Tagger, models</p>
 *
 * @author David B. Bracewell
 */
public class StructuredPerceptronLearner extends SequenceLabelerLearner {
   private static final Logger log = Logger.getLogger(StructuredPerceptronLearner.class);
   private static final long serialVersionUID = 1209076471049751899L;
   private int maxIterations = 10;
   private double tolerance = 0.00001;
   private NDArray[] cWeights;

   /**
    * Gets max iterations.
    *
    * @return the max iterations
    */
   public int getMaxIterations() {
      return maxIterations;
   }

   /**
    * Sets max iterations.
    *
    * @param maxIterations the max iterations
    */
   public void setMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
   }

   /**
    * Gets tolerance.
    *
    * @return the tolerance
    */
   public double getTolerance() {
      return tolerance;
   }

   /**
    * Sets tolerance.
    *
    * @param tolerance the tolerance
    */
   public void setTolerance(double tolerance) {
      this.tolerance = tolerance;
   }

   @Override
   public void resetLearnerParameters() {
      cWeights = null;
   }

   @Override
   protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
      StructuredPerceptron model = new StructuredPerceptron(this);
      model.setDecoder(getDecoder());

      int nC = model.numberOfLabels();

      model.weights = new NDArray[nC];
      cWeights = new NDArray[nC];
      for (int i = 0; i < nC; i++) {
         model.weights[i] = NDArrayFactory.SPARSE_FLOAT.zeros(model.numberOfFeatures());
         cWeights[i] = NDArrayFactory.SPARSE_FLOAT.zeros(model.numberOfFeatures());
      }


      double oldOldError = 0;
      double oldError = 0;
      final DecimalFormat formatter = new DecimalFormat("###.00%");

      List<Sequence> sequenceList = Lists.newLinkedList(Collect.asIterable(dataset.iterator()));
      int c = 1;
      for (int itr = 0; itr < maxIterations; itr++) {
         Stopwatch sw = Stopwatch.createStarted();

         double count = 0;
         double correct = 0;

         for (Sequence sequence : sequenceList) {

            Labeling lblResult = model.label(sequence);

            double diff = 0;
            for (Context<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
               count++;
               if (!iterator.next().getLabel().equals(lblResult.getLabel(iterator.getIndex()))) {
                  diff++;
               } else {
                  correct++;
               }
            }


            if (diff > 0) {
               for (Context<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
                  Instance instance = iterator.next();
                  int y = (int) model.getLabelEncoder().encode(instance.getLabel());
                  int yHat = (int) model.getLabelEncoder().encode(lblResult.getLabel(iterator.getIndex()));
                  if (y != yHat) {
                     for (Feature feature : instance) {
                        int fid = (int) model.getFeatureEncoder().encode(feature.getFeatureName());
                        model.weights[yHat].decrement(fid);
                        model.weights[y].increment(fid);
                        cWeights[yHat].decrement(fid);
                        cWeights[y].increment(fid);
                     }
                     for (String feature : Collect.asIterable(
                        transitionFeatures.extract(lblResult.iterator(sequence, iterator.getIndex())))) {
                        int fid = (int) model.getFeatureEncoder().encode(feature);
                        model.weights[yHat].decrement(fid);
                        cWeights[yHat].decrement(fid);
                     }
                     for (String feature : Collect.asIterable(transitionFeatures.extract(iterator))) {
                        int fid = (int) model.getFeatureEncoder().encode(feature);
                        model.weights[y].increment(fid);
                        cWeights[y].increment(fid);
                     }
                  }
               }

               c++;
            }

         }

         sw.stop();
         log.info("iteration={0} accuracy={1} ({2}/{3}) [completed in {4}]", itr + 1, formatter.format(correct / count),
                  correct, count, sw);

         if (count - correct == 0) {
            break;
         }

         double error = (count - correct) / count;
         if (itr > 2 && Math.abs(error - oldError) < tolerance && Math.abs(error - oldOldError) < tolerance) {
            break;
         }

         oldOldError = oldError;
         oldError = error;
      }

      final double C = c;
      for (int ci = 0; ci < nC; ci++) {
         NDArray v = model.weights[ci];
         cWeights[ci].forEachSparse(entry -> v.decrement(entry.getIndex(), entry.getValue() / C));
      }

      return model;
   }


}// END OF StructuredPerceptronLearner

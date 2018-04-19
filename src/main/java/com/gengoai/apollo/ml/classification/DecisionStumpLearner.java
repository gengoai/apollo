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

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.stat.measure.ContingencyTable;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;

import java.util.List;

/**
 * <p>Learner for decision stumps, or Zero-1 Rules. Acts as weak learner for ensemble techniques like {@link
 * BaggingLearner}</p>
 *
 * @author David B. Bracewell
 */
public class DecisionStumpLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      DecisionStump stump = new DecisionStump(this);

      int bestIndex = -1;
      double bestSplit = 0;
      double bestScore = Double.POSITIVE_INFINITY;
      double[] bestLowerDistribution = new double[stump.numberOfLabels()];
      double[] bestUpperDistribution = new double[stump.numberOfLabels()];
      double[] totalLabelCounts = new double[stump.numberOfLabels()];

      for (Instance instance : dataset) {
         totalLabelCounts[(int) stump.encodeLabel(instance.getLabel())]++;
      }


      for (int featureID = 0; featureID < stump.numberOfFeatures(); featureID++) {
         ContingencyTable counts = new ContingencyTable(2, stump.numberOfLabels());
         for (int i = 0; i < totalLabelCounts.length; i++) {
            counts.set(1, i, totalLabelCounts[i]);
         }
         final String featureName = stump.decodeFeature(featureID).toString();
         List<Instance> sorted = dataset.stream()
                                        .sorted(true, ii -> ii.getValue(featureName))
                                        .collect();
         for (int iid = 0; iid < sorted.size() - 1; iid++) {
            Instance current = sorted.get(iid);
            int currentLabel = (int) stump.encodeLabel(current.getLabel());
            double currentFeatureValue = current.getValue(featureName);
            counts.set(0, currentLabel, counts.get(0, currentLabel) + 1);
            counts.set(1, currentLabel, counts.get(1, currentLabel) - 1);
            Instance next = sorted.get(iid + 1);

            if (bestIndex == -1) {
               bestIndex = featureID;
               bestSplit = currentFeatureValue;
               for (int i = 0; i < stump.numberOfLabels(); i++) {
                  bestLowerDistribution[i] = counts.get(0, i);
                  bestUpperDistribution[i] = counts.get(1, i);
               }
            }

            if (currentFeatureValue < next.getValue(featureName)) {
               double score = -Math.log(counts.rowSum(0)) - Math.log(counts.rowSum(1));
               score = (score + Math.log(counts.getSum())) / (counts.getSum() + 2);
               if (score < bestScore) {
                  bestIndex = featureID;
                  bestSplit = currentFeatureValue;
                  bestScore = score;
                  for (int i = 0; i < stump.numberOfLabels(); i++) {
                     bestLowerDistribution[i] = counts.get(0, i);
                     bestUpperDistribution[i] = counts.get(1, i);
                  }
               }

            }
         }
      }

      stump.featureId = bestIndex;
      stump.featureValue = bestSplit;
      stump.lowerDecision = bestLowerDistribution;
      stump.upperDecision = bestUpperDistribution;
      return stump;
   }
}//END OF DecisionStumpLearner

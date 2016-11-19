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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

/**
 * <p>A decision stump, or Zero-One rule, classifier which makes its classification based on the value of a single
 * feature. Acts as weak learner for ensemble techniques like {@link BaggingLearner}</p>
 *
 * @author David B. Bracewell
 */
public class DecisionStump extends Classifier {
   private static final long serialVersionUID = 1L;
   int featureId;
   double featureValue;
   double[] lowerDecision;
   double[] upperDecision;

   /**
    * Instantiates a new Decision Stump Classifier.
    *
    * @param encoderPair   the pair of encoders to convert feature names into int/double values
    * @param preprocessors the preprocessors that the classifier will need apply at runtime
    */
   protected DecisionStump(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }

   @Override
   public Classification classify(@NonNull Vector vector) {
      double[] distribution;
      if (vector.get(featureId) > featureValue) {
         distribution = upperDecision;
      } else {
         distribution = lowerDecision;
      }
      return createResult(distribution);
   }
}//END OF DecisionStump

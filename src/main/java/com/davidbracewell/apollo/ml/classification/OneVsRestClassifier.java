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

/**
 * <p>Classifier tha employs a one-vs-rest strategy of combining binary classifiers to produce a multi-class
 * classifier.</p>
 *
 * @author David B. Bracewell
 */
public class OneVsRestClassifier extends Classifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Classifiers.
    */
   Classifier[] classifiers;
   /**
    * The Normalize.
    */
   boolean normalize = false;

   protected OneVsRestClassifier(ClassifierLearner learner) {
      super(learner);
   }


   @Override
   public Classification classify(NDArray vector) {
      NDArray distribution = NDArrayFactory.wrap(new double[numberOfLabels()]);
      for (int ci = 0; ci < distribution.length(); ci++) {
         distribution.set(ci, classifiers[ci].classify(vector).distribution()[1]);
      }
      if (normalize) {
         //Softmax normalization and log normalization
         distribution.subi(distribution.max())
                     .expi();
         distribution.divi(distribution.sum())
                     .logi();
      }
      return new Classification(distribution.toArray(), getLabelEncoder());
   }

}//END OF OneVsRestClassifier

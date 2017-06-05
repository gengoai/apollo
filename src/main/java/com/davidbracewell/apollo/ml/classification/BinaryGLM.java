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
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.activation.LinearActivation;
import lombok.NonNull;

/**
 * <p>A binary class generalized linear model.</p>
 *
 * @author David B. Bracewell
 */
public class BinaryGLM extends Classifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Weights.
    */
   Vector weights;
   /**
    * The Bias.
    */
   double bias;

   Activation activation = new LinearActivation();

   /**
    * Instantiates a new Classifier.
    *
    * @param encoderPair   the encoder pair
    * @param preprocessors the preprocessors
    */
   protected BinaryGLM(EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
      super(encoderPair, preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      double[] dist = new double[2];
      dist[1] = activation.apply(weights.dot(vector) + bias);
      if (activation.isProbabilistic()) {
         dist[0] = 1d - dist[1];
      } else {
         dist[0] = -dist[1];
      }
      return createResult(dist);
   }

}//END OF BinaryGLM

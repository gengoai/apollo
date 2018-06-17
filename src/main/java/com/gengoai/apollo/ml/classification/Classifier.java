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

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.MultiCounter;
import lombok.Getter;
import lombok.NonNull;

/**
 * Base class for classifiers that predicts the label, or class, for a set of features.
 *
 * @author David B. Bracewell
 */
public abstract class Classifier implements Model {
   private static final long serialVersionUID = 1L;
   @Getter
   private final PreprocessorList<Instance> preprocessors;
   private final EncoderPair encoderPair;

   protected Classifier(@NonNull ClassifierLearner learner) {
      this.preprocessors = learner.getPreprocessors().getModelProcessors();
      this.encoderPair = learner.getEncoderPair();
   }

   protected Classifier(@NonNull PreprocessorList<Instance> preprocessors, EncoderPair encoderPair){
      this.preprocessors = preprocessors.getModelProcessors();
      this.encoderPair = encoderPair;
   }

   /**
    * Predicts the label, or class, of the given instance
    *
    * @param instance the instance whose class we want to predict
    * @return the classification result
    */
   public Classification classify(@NonNull Instance instance) {
      return classify(preprocessors.apply(instance).toVector(encoderPair));
   }

   /**
    * Predicts the label, or class, of the given vector. Note, that all preprocessing must already be performed on the
    * vector.
    *
    * @param vector the vector whose class we want to predict
    * @return the classification result
    */
   public abstract Classification classify(NDArray vector);

   /**
    * Convenience method for creating classification results.
    *
    * @param distribution the distribution of probabilities
    * @return the classification result
    */
   protected Classification createResult(double[] distribution) {
      return new Classification(distribution, getLabelEncoder());
   }

   protected Classification createResult(Counter<String> distribution) {
      double[] dis = new double[getLabelEncoder().size()];
      distribution.divideBySum();
      distribution.forEach((i, v) -> dis[(int) encodeLabel(i)] = v);
      return new Classification(dis, getLabelEncoder());
   }

   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   /**
    * Gets the parameters of the model. Typically these will be Feature-Class-Weight triplets.
    *
    * @return the model parameters
    */
   public MultiCounter<String, String> getModelParameters() {
      throw new UnsupportedOperationException();
   }

   @Override
   public int numberOfFeatures() {
      return encoderPair.numberOfFeatures();
   }

   protected Instance preprocess(Instance instance) {
      return preprocessors.apply(instance);
   }

}//END OF Classifier

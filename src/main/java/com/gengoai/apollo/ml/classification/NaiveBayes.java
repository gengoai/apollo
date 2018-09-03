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
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.collection.counter.HashMapMultiCounter;
import com.gengoai.collection.counter.MultiCounter;
import lombok.NonNull;

/**
 * Naive Bayes model specifically designed for text classification problems
 *
 * @author David B. Bracewell
 */
public class NaiveBayes extends Classifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Model type.
    */
   protected ModelType modelType;
   /**
    * The Priors.
    */
   protected double[] priors;
   /**
    * The Conditionals.
    */
   protected double[][] conditionals;

   /**
    * Instantiates a new Naive bayes.
    *
    * @param learner the learner
    */
   public NaiveBayes(ClassifierLearner learner) {
      this(learner, ModelType.Bernoulli);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param learner   the learner
    * @param modelType the model type
    */
   public NaiveBayes(ClassifierLearner learner, @NonNull ModelType modelType) {
      super(learner);
      this.modelType = modelType;
   }

   @Override
   public Classification classify(@NonNull NDArray instance) {
      return createResult(modelType.distribution(instance, priors, conditionals));
   }

   @Override
   public MultiCounter<String, String> getModelParameters() {
      MultiCounter<String, String> weights = new HashMapMultiCounter<>();
      for (int fi = 0; fi < numberOfFeatures(); fi++) {
         String featureName = getFeatureEncoder().decode(fi).toString();
         for (int ci = 0; ci < numberOfLabels(); ci++) {
            weights.set(featureName, getLabelEncoder().decode(ci).toString(), conditionals[fi][ci]);
         }
      }
      return weights;
   }

   /**
    * Gets the type of Naive Bayes model being used.
    *
    * @return the model type
    */
   public ModelType getModelType() {
      return modelType;
   }

   /**
    * Three types of Naive Bayes models are supported each of which have their own potential pros and cons and may work
    * better or worse for different types of data.
    */
   public enum ModelType {
      /**
       * Multinomial Naive Bayes using Laplace Smoothing
       */
      Multinomial,
      /**
       * Bernoulli Naive Bayes where each feature is treated as being binary
       */
      Bernoulli {
         @Override
         double convertValue(double value) {
            return value > 0 ? 1.0 : 0.0;
         }

         @Override
         double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
            return (conditionalCount + 1) / (priorCount + 2);
         }

         @Override
         double[] distribution(NDArray instance, double[] priors, double[][] conditionals) {
            NDArray distribution = NDArrayFactory.columnVector(priors);
            for (int i = 0; i < priors.length; i++) {
               for (int f = 0; f < conditionals.length; f++) {
                  if (instance.get(f) != 0) {
                     distribution.increment(i, Math.log(conditionals[f][i]));
                  } else {
                     distribution.increment(i, Math.log(1 - conditionals[f][i]));
                  }
               }
            }
            distribution.mapi(Math::exp);
            return distribution.toDoubleArray();
         }
      },
      /**
       * Complementary Naive Bayes which works similarly to the Multinomial version, but is trained differently to
       * better handle label imbalance.
       */
      Complementary;

      /**
       * Converts a features value.
       *
       * @param value the value
       * @return the converted value
       */
      double convertValue(double value) {
         return value;
      }

      /**
       * Calculates a distribution of probabilities over the labels given a vector instance and the model priors and
       * conditionals.
       *
       * @param instance     the instance to calculate the distribution for
       * @param priors       the label priors
       * @param conditionals the feature-label conditional probabilities
       * @return the distribution as an array
       */
      double[] distribution(NDArray instance, double[] priors, double[][] conditionals) {
         NDArray distribution = NDArrayFactory.columnVector(priors);
         instance.forEachSparse(entry -> {
            for (int i = 0; i < priors.length; i++) {
               distribution.decrement(i, entry.getValue() * conditionals[(int) entry.getIndex()][i]);
            }
         });
         return distribution.mapi(Math::exp).toDoubleArray();
      }

      /**
       * Normalizes (smooths) the conditional probability given the conditional count, prior count, total label count,
       * and vocabulary size.
       *
       * @param conditionalCount the conditional count
       * @param priorCount       the prior count
       * @param totalLabelCount  the total label count
       * @param V                the vocabulary size
       * @return the normalized  (smoothed) conditional probability
       */
      double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
         return (conditionalCount + 1) / (totalLabelCount + V);
      }

   }
}//END OF NaiveBayes

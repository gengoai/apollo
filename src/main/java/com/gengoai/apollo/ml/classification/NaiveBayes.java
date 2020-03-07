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

package com.gengoai.apollo.ml.classification;

import com.gengoai.ParameterDef;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.data.VectorizedDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import com.gengoai.math.Math2;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

/**
 * <p>Naive Bayes model specifically designed for text classification problems. Allows for training the following type
 * of models:</p>
 * <ul>
 * <li>Multinomial</li>
 * <li>Bernoulli</li>
 * <li>Complementary</li>
 * </ul>
 *
 * @author David B. Bracewell
 */
public class NaiveBayes extends Classifier {
   private static final long serialVersionUID = 1L;
   private double[][] conditionals;
   private ModelType modelType;
   private double[] priors;

   /**
    * Instantiates a new Naive bayes.
    *
    * @param preprocessors the preprocessors
    */
   public NaiveBayes(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Naive bayes.
    *
    * @param modelParameters the model parameters
    */
   public NaiveBayes(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   protected void fitPreprocessed(ExampleDataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = Cast.as(fitParameters);
      conditionals = new double[getNumberOfFeatures()][getNumberOfLabels()];
      priors = new double[getNumberOfLabels()];
      modelType = parameters.modelType.value();
      double[] labelCounts = new double[getNumberOfLabels()];

      double N = 0;
      for (Example instance : preprocessed) {
         if (instance.hasLabel()) {
            N++;
            NDArray vector = instance.transform(getPipeline());
            int ci = (int) vector.getLabelAsNDArray().argmax();
            priors[ci] += instance.getWeight();
            vector.forEachSparse((index, value) -> {
               labelCounts[ci] += value;
               conditionals[(int) index][ci] += instance.getWeight() * modelType.convertValue(value);
            });
         }
      }

      double V = getNumberOfFeatures();
      for (int featureIndex = 0; featureIndex < conditionals.length; featureIndex++) {
         double[] tmp = Arrays.copyOf(conditionals[featureIndex], conditionals[featureIndex].length);
         for (int labelIndex = 0; labelIndex < priors.length; labelIndex++) {
            if (modelType == NaiveBayes.ModelType.Complementary) {
               double nCi = 0;
               double nC = 0;
               for (int j = 0; j < priors.length; j++) {
                  if (j != labelIndex) {
                     nCi += tmp[j];
                     nC += labelCounts[j];
                  }
               }

               conditionals[featureIndex][labelIndex] = Math2.safeLog(
                  modelType.normalize(nCi, priors[labelIndex], nC, V));
            } else {
               conditionals[featureIndex][labelIndex] = Math2.safeLog(
                  modelType.normalize(conditionals[featureIndex][labelIndex], priors[labelIndex],
                                      labelCounts[labelIndex], V));
            }
         }

      }

      for (int i = 0; i < priors.length; i++) {
         priors[i] = Math.log(priors[i] / N);
      }

   }

   @Override
   public void fit(VectorizedDataset dataset, FitParameters<?> fitParameters) {
      throw new UnsupportedOperationException();
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(modelType.distribution(example.preprocessAndTransform(getPipeline()),
                                                       priors,
                                                       conditionals),
                                getPipeline().labelVectorizer);
   }

   @Override
   public Classification predict(NDArray example) {
      throw new UnsupportedOperationException();
   }

   /**
    * Three types of Naive Bayes  are supported each of which have their own potential pros and cons and may work better
    * or worse for different types of data.
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
         NDArray distribution(NDArray instance, double[] priors, double[][] conditionals) {
            NDArray distribution = NDArrayFactory.ND.columnVector(priors);
            for (int i = 0; i < priors.length; i++) {
               for (int f = 0; f < conditionals.length; f++) {
                  double value = distribution.get(i);
                  if (instance.get(f) != 0) {
                     distribution.set(i, value + Math2.safeLog(conditionals[f][i]));
                  } else {
                     distribution.set(i, value + Math2.safeLog(1 - conditionals[f][i]));
                  }
               }
            }
            distribution.mapi(FastMath::exp);
            return distribution;
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
       * Calculates a distribution of probabilities over the labels given a vector instance and the priors and
       * conditionals.
       *
       * @param instance     the instance to calculate the distribution for
       * @param priors       the label priors
       * @param conditionals the feature-label conditional probabilities
       * @return the distribution as an array
       */
      NDArray distribution(NDArray instance, double[] priors, double[][] conditionals) {
         NDArray distribution = NDArrayFactory.ND.columnVector(priors);
         instance.forEachSparse((index, value) -> {
            for (int i = 0; i < priors.length; i++) {
               distribution.set(i, distribution.get(i) - value * conditionals[(int) index][i]);
            }

         });
         return distribution.mapi(Math::exp);
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

   public static final ParameterDef<ModelType> modelTypeParam = ParameterDef.param("modelType", ModelType.class);

   /**
    * Custom {@link FitParameters} for Naive Bayes.
    */
   public static class Parameters extends FitParameters<Parameters> {
      /**
       * The type of Naive Bayes model to train.
       */
      public final Parameter<ModelType> modelType = parameter(modelTypeParam, ModelType.Bernoulli);
   }
}//END OF NaiveBayes

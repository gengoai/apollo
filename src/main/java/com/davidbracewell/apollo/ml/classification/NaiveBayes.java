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

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.HashMapMultiCounter;
import com.davidbracewell.collection.counter.MultiCounter;
import lombok.NonNull;

/**
 * The type Naive bayes.
 *
 * @author David B. Bracewell
 */
public class NaiveBayes extends Classifier {
  private static final long serialVersionUID = 1L;
  /**
   * The Model type.
   */
  ModelType modelType;
  /**
   * The Priors.
   */
  double[] priors;
  /**
   * The Conditionals.
   */
  double[][] conditionals;

  public NaiveBayes(EncoderPair encoderPair, PreprocessorList<Instance> preprocessors) {
    this(encoderPair, preprocessors, ModelType.Bernoulli);
  }

  public NaiveBayes(@NonNull EncoderPair encoderPair, PreprocessorList<Instance> preprocessors, ModelType modelType) {
    super(encoderPair, preprocessors);
    this.modelType = modelType;
  }

  /**
   * Gets model type.
   *
   * @return the model type
   */
  public ModelType getModelType() {
    return modelType;
  }

  @Override
  public Classification classify(@NonNull Vector instance) {
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
   * The enum Model type.
   */
  public enum ModelType {
    /**
     * Multinomial model type.
     */
    Multinomial,
    /**
     * Bernoulli model type.
     */
    Bernoulli {
      @Override
      double convertValue(double value) {
        return 1.0;
      }

      @Override
      double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
        return (conditionalCount + 1) / (priorCount + 2);
      }

      @Override
      double[] distribution(Vector instance, double[] priors, double[][] conditionals) {
        DenseVector distribution = new DenseVector(priors);
        for (int i = 0; i < priors.length; i++) {
          for (int f = 0; f < conditionals.length; f++) {
            if (instance.get(f) != 0) {
              distribution.increment(i, Math.log(conditionals[f][i]));
            } else {
              distribution.increment(i, Math.log(1 - conditionals[f][i]));
            }
          }
        }
        distribution.mapSelf(Math::exp);
        return distribution.toArray();
      }
    },
    /**
     * Complementary model type.
     */
    Complementary;

    double convertValue(double value) {
      return value;
    }

    double normalize(double conditionalCount, double priorCount, double totalLabelCount, double V) {
      return (conditionalCount + 1) / (totalLabelCount + V);
    }

    double[] distribution(Vector instance, double[] priors, double[][] conditionals) {
      DenseVector distribution = new DenseVector(priors);
      instance.forEachSparse(entry -> {
        for (int i = 0; i < priors.length; i++) {
          distribution.decrement(i, entry.getValue() * conditionals[entry.getIndex()][i]);
        }
      });
      return distribution.mapSelf(Math::exp).toArray();
    }

  }
}//END OF NaiveBayes

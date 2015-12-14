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

package com.davidbracewell.apollo.ml.classification.bayes;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.MultiCounter;
import com.davidbracewell.collection.MultiCounters;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;

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
  final ModelType modelType;
  /**
   * The Priors.
   */
  double[] priors;
  /**
   * The Conditionals.
   */
  double[][] conditionals;


  /**
   * Instantiates a new Classifier.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   * @param preprocessors  the preprocessors
   * @param modelType      the model type
   */
  protected NaiveBayes(IndexEncoder labelEncoder, Encoder featureEncoder, PreprocessorList<Instance> preprocessors, ModelType modelType) {
    super(labelEncoder, featureEncoder, preprocessors);
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
  public ClassifierResult classify(@NonNull Vector instance) {
    switch (modelType) {
      case Bernoulli:
        return bernoulli(instance);
      case Multinomial:
        return multinomial(instance);
      case Complementary:
        return complementary(instance);
    }
    throw new IllegalStateException(modelType + " is not valid");
  }

  private ClassifierResult bernoulli(Vector instance) {
    DenseVector distribution = new DenseVector(priors);
    for (int i = 0; i < numberOfLabels(); i++) {
      for (int f = 0; f < numberOfFeatures(); f++) {
        if (instance.get(f) != 0) {
          distribution.increment(i, FastMath.log(conditionals[f][i]));
        } else {
          distribution.increment(i, FastMath.log(1 - conditionals[f][i]));
        }
      }
    }
    distribution.mapDivideSelf(distribution.sum());
    distribution.mapSelf(d -> 1.0 - d);
    return new ClassifierResult(distribution.toArray(), getLabelEncoder());
  }

  private ClassifierResult multinomial(Vector instance) {
    DenseVector distribution = new DenseVector(priors);
    instance.forEachSparse(entry -> {
      for (int i = 0; i < numberOfLabels(); i++) {
        distribution.increment(i, entry.getValue() * conditionals[entry.getIndex()][i]);
      }
    });
    distribution.mapDivideSelf(distribution.sum());
    distribution.mapSelf(d -> 1.0 - d);
    return new ClassifierResult(distribution.toArray(), getLabelEncoder());
  }

  private ClassifierResult complementary(Vector instance) {
    DenseVector distribution = new DenseVector(priors);
    instance.forEachSparse(entry -> {
      for (int i = 0; i < numberOfLabels(); i++) {
        distribution.decrement(i, entry.getValue() * conditionals[entry.getIndex()][i]);
      }
    });
    distribution.mapDivideSelf(distribution.sum());
    distribution.mapSelf(d -> 1.0 - d);
    return new ClassifierResult(distribution.toArray(), getLabelEncoder());
  }

  @Override
  public MultiCounter<String, String> getModelParameters() {
    MultiCounter<String, String> weights = MultiCounters.newHashMapMultiCounter();
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
    Bernoulli,
    /**
     * Complementary model type.
     */
    Complementary
  }
}//END OF NaiveBayes

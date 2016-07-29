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
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.MultiCounter;
import com.davidbracewell.conversion.Cast;
import com.google.common.base.Preconditions;
import lombok.NonNull;

/**
 * The interface Classifier.
 *
 * @author David B. Bracewell
 */
public abstract class Classifier implements Model {
  private static final long serialVersionUID = 1L;
  private final PreprocessorList<Instance> preprocessors;
  private Featurizer<Object> featurizer;
  private EncoderPair encoderPair;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected Classifier(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    this.encoderPair = encoderPair;
    Preconditions.checkArgument(encoderPair.getLabelEncoder() instanceof IndexEncoder, "Classifiers only allow IndexEncoders for labels.");
    this.preprocessors = preprocessors.getModelProcessors();
  }

  /**
   * Classifier result classifier result.
   *
   * @param input the input
   * @return the classifier result
   */
  @SuppressWarnings("unchecked")
  public Classification classify(@NonNull Object input) {
    Preconditions.checkNotNull(featurizer, "Featurizer has not been set on the classifier");
    return classify(featurizer.extract(input));
  }


  /**
   * Classify classifier result.
   *
   * @param instance the instance
   * @return the classifier result
   */
  public final Classification classify(@NonNull Instance instance) {
    return classify(preprocessors.apply(instance).toVector(getEncoderPair()));
  }

  /**
   * Classify classifier result.
   *
   * @param vector the vector
   * @return the classifier result
   */
  public abstract Classification classify(Vector vector);

  /**
   * Sets featurizer.
   *
   * @param featurizer the featurizer
   */
  public void setFeaturizer(Featurizer<?> featurizer) {
    this.featurizer = Cast.as(featurizer);
  }

  /**
   * Gets model parameters.
   *
   * @return the model parameters
   */
  public MultiCounter<String, String> getModelParameters() {
    throw new UnsupportedOperationException();
  }

  /**
   * Create result classification.
   *
   * @param distribution the distribution
   * @return the classification
   */
  protected Classification createResult(double[] distribution) {
    return new Classification(distribution, getLabelEncoder());
  }

  @Override
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

}//END OF Classifier

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

package com.davidbracewell.apollo.learning.classification;

import com.davidbracewell.apollo.learning.FeatureEncoder;
import com.davidbracewell.apollo.learning.Featurizer;
import com.davidbracewell.apollo.learning.Instance;
import com.davidbracewell.collection.Index;
import com.davidbracewell.io.resource.Resource;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The interface Classifier.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class Classifier<T> implements Serializable {
  private static final long serialVersionUID = 1L;

  private final Index<String> classLabels;
  private final FeatureEncoder featureEncoder;
  private final Featurizer<T> featurizer;

  /**
   * Instantiates a new Classifier.
   *
   * @param classLabels    the class labels
   * @param featureEncoder the feature encoder
   * @param featurizer     the featurizer
   */
  protected Classifier(Index<String> classLabels, FeatureEncoder featureEncoder, Featurizer<T> featurizer) {
    this.classLabels = classLabels;
    this.featureEncoder = featureEncoder;
    this.featurizer = featurizer;
  }

  /**
   * Gets class labels.
   *
   * @return the class labels
   */
  public Index<String> getClassLabels() {
    return classLabels;
  }

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  public FeatureEncoder getFeatureEncoder() {
    return featureEncoder;
  }

  /**
   * Classify classifier result.
   *
   * @param instance the instance
   * @return the classifier result
   */
  public abstract ClassifierResult classify(Instance instance);

  /**
   * Classify classifier result.
   *
   * @param input the input
   * @return the classifier result
   */
  public final ClassifierResult classify(@NonNull T input) {
    return classify(Instance.create(featurizer.apply(input)));
  }

  /**
   * Read model classifier.
   *
   * @param <T>           the type parameter
   * @param modelResource the model resource
   * @return the classifier
   * @throws Exception the exception
   */
  public static <T> Classifier<T> readModel(@NonNull Resource modelResource) throws Exception {
    return modelResource.readObject();
  }

  /**
   * Write model.
   *
   * @param modelResource the model resource
   * @throws Exception the exception
   */
  public final void writeModel(@NonNull Resource modelResource) throws Exception {
    modelResource.writeObject(this);
  }

}//END OF Classifier

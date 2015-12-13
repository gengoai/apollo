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

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.List;

/**
 * The interface Classifier learner.
 *
 * @author David B. Bracewell
 */
public abstract class ClassifierLearner extends Learner<Instance, Classifier> {
  private static final long serialVersionUID = 1L;

//  /**
//   * Train classifier.
//   *
//   * @param vectors      the vectors
//   * @param labelEncoder the label encoder
//   * @return the classifier
//   */
//  public final Classifier train(@NonNull List<FeatureVector> vectors, @NonNull Encoder labelEncoder) {
//    Preconditions.checkArgument(vectors.size() > 0, "Must have at least one vector");
//    Classifier model = trainImpl(
//      vectors,
//      labelEncoder,
//      vectors.get(0).getFeatureEncoder()
//    );
//    model.finishTraining();
//    return model;
//  }
//
//  /**
//   * Train classifier.
//   *
//   * @param vectors        the vectors
//   * @param labelEncoder   the label encoder
//   * @param featureEncoder the feature encoder
//   * @return the classifier
//   */
//  protected abstract Classifier trainImpl(@NonNull List<FeatureVector> vectors, @NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder);

}//END OF ClassifierLearner

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

package com.davidbracewell.apollo.ml.classification.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class BinaryGLM extends Classifier {
  private static final long serialVersionUID = 1L;
  Vector weights;
  double bias;

  /**
   * Instantiates a new Classifier.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   * @param preprocessors  the preprocessors
   */
  protected BinaryGLM(Encoder labelEncoder, Encoder featureEncoder, @NonNull PreprocessorList<Instance> preprocessors) {
    super(labelEncoder, featureEncoder, preprocessors);
  }

  @Override
  public ClassifierResult classify(Vector vector) {
    double[] dist = new double[2];
    dist[1] = weights.dot(vector) + bias;
    dist[0] = -dist[1];
    return new ClassifierResult(dist, getLabelEncoder());
  }

}//END OF BinaryGLM

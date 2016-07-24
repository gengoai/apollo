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

package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceValidator;
import com.davidbracewell.apollo.ml.sequence.TransitionFeatures;
import com.google.common.collect.Lists;

import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class CRF extends SequenceLabeler {
  private static final long serialVersionUID = 1L;
  Vector[] weights;
  double scale = 0;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder       the label encoder
   * @param featureEncoder     the feature encoder
   * @param preprocessors      the preprocessors
   * @param transitionFeatures the transition features
   * @param validator
   */
  public CRF(LabelEncoder labelEncoder, Encoder featureEncoder, PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures, SequenceValidator validator) {
    super(labelEncoder, featureEncoder, preprocessors, transitionFeatures, validator);
  }

  public CRF copy() {
    CRF copy = new CRF(
      getLabelEncoder(),
      getFeatureEncoder(),
      getPreprocessors(),
      getTransitionFeatures(),
      getValidator()
    );
    copy.scale = this.scale;
    copy.weights = new Vector[this.weights.length];
    for (int i = 0; i < this.weights.length; i++) {
      copy.weights[i] = this.weights[i].copy();
    }
    return copy;
  }

  @Override
  public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
    List<Feature> features = Lists.newArrayList(observation);
    while (transitions.hasNext()) {
      features.add(Feature.TRUE(transitions.next()));
    }
    Vector instance = Instance.create(features).toVector(getEncoderPair());
    double[] dist = new double[weights.length];
    for (int i = 0; i < dist.length; i++) {
      dist[i] = weights[i].dot(instance) * scale;
    }
    return dist;
  }

  void rescale() {
    if (scale != 1.0) {
      for (Vector v : weights)
        v.mapMultiplySelf(scale);
      scale = 1;
    }
  }

  double mag() {
    double norm = 0;
    for (int i = 0; i < weights.length; i++) {
      norm += weights[i].dot(weights[i]);
    }
    return norm * scale * scale;
  }

}//END OF Crf

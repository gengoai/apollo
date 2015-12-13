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
import com.google.common.collect.Lists;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import lombok.NonNull;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier {
  private static final long serialVersionUID = 1L;
  Model model;

  /**
   * Instantiates a new Classifier.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   * @param preprocessors
   */
  protected LibLinearModel(Encoder labelEncoder, Encoder featureEncoder, @NonNull PreprocessorList<Instance> preprocessors) {
    super(labelEncoder, featureEncoder, preprocessors);
  }

  public static Feature[] toFeature(Vector vector) {
    List<Vector.Entry> entries = Lists.newArrayList(vector.orderedNonZeroIterator());
    Feature[] feature = new Feature[entries.size()];
    for (int i = 0; i < entries.size(); i++) {
      feature[i] = new FeatureNode(entries.get(i).index+1, entries.get(i).value);
    }
    return feature;
  }

  @Override
  public ClassifierResult classify(Vector vector) {
    double[] p = new double[numberOfLabels()];
    if (model.isProbabilityModel()) {
      Linear.predictProbability(model, toFeature(vector), p);
    } else {
      Linear.predictValues(model, toFeature(vector), p);
    }

    //re-arrange the probabilities to match the target feature
    double[] prime = new double[numberOfLabels()];
    int[] labels = model.getLabels();
    for (int i = 0; i < labels.length; i++) {
      prime[labels[i]] = p[i];
    }

    return new ClassifierResult(prime, getLabelEncoder());
  }

}//END OF LibLinearModel

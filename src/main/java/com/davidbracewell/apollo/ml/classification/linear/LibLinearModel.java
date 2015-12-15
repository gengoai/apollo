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
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.MultiCounter;
import com.davidbracewell.collection.MultiCounters;
import com.google.common.collect.Lists;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import lombok.NonNull;

import java.util.List;

/**
 * The type Lib linear model.
 *
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier {
  private static final long serialVersionUID = 1L;
  /**
   * The Model.
   */
  Model model;

  /**
   * Instantiates a new Classifier.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  protected LibLinearModel(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    super(encoderPair, preprocessors);
  }

  /**
   * To feature feature [ ].
   *
   * @param vector the vector
   * @return the feature [ ]
   */
  public static Feature[] toFeature(Vector vector) {
    List<Vector.Entry> entries = Lists.newArrayList(vector.orderedNonZeroIterator());
    Feature[] feature = new Feature[entries.size()];
    for (int i = 0; i < entries.size(); i++) {
      feature[i] = new FeatureNode(entries.get(i).index + 1, entries.get(i).value);
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

    return createResult(prime);
  }

  @Override
  public MultiCounter<String, String> getModelParameters() {
    double[] modelWeights = model.getFeatureWeights();
    int[] labels = model.getLabels();
    MultiCounter<String, String> weights = MultiCounters.newHashMapMultiCounter();

    int nrClass = model.getNrClass() - 1;
    for (int i = 0, index = 0; i < model.getNrFeature(); i++, index += nrClass) {
      String featureName = getFeatureEncoder().decode(i).toString();
      for (int j = 0; j < nrClass; j++) {
        weights.set(featureName, getLabelEncoder().decode(labels[j]).toString(), modelWeights[j + index]);
      }
    }

    if (modelWeights.length > (nrClass * model.getNrFeature())) {
      for (int j = modelWeights.length - nrClass, ci = 0; j < modelWeights.length; j++, ci++) {
        weights.set("**BIAS**", getLabelEncoder().decode(labels[ci]).toString(), modelWeights[j]);
      }
    }

    return weights;
  }

}//END OF LibLinearModel

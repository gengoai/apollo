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

package com.davidbracewell.apollo.learning.classification.bayes;

import com.davidbracewell.apollo.learning.FeatureEncoder;
import com.davidbracewell.apollo.learning.Featurizer;
import com.davidbracewell.apollo.learning.Instance;
import com.davidbracewell.apollo.learning.classification.ClassifierResult;
import com.davidbracewell.collection.Index;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class BernoulliNaiveBayes<T> extends NaiveBayes<T> {
  private static final long serialVersionUID = 1L;

  /**
   * Instantiates a new Classifier.
   *
   * @param classLabels    the class labels
   * @param featureEncoder the feature encoder
   */
  protected BernoulliNaiveBayes(Index<String> classLabels, FeatureEncoder featureEncoder, Featurizer<T> featurizer) {
    super(classLabels, featureEncoder, featurizer);
  }


  @Override
  public ClassifierResult classify(Instance instance) {
    int numClasses = getClassLabels().size();
    double[] probabilities = new double[numClasses];

    for (int i = 0; i < numClasses; i++) {
      probabilities[i] = FastMath.log(priors[i]);
      for (int f = 0; f < getFeatureEncoder().size(); f++) {
        double value = instance.getValue(getFeatureEncoder().decode(f));
        if (value != 0) {
          probabilities[i] += FastMath.log(conditionals[f][i]);
        } else {
          probabilities[i] += FastMath.log(1 - conditionals[f][i]);
        }
      }
      probabilities[i] = Math.exp(probabilities[i]);
    }
    return null;
  }


}//END OF BernoulliNaiveBayes

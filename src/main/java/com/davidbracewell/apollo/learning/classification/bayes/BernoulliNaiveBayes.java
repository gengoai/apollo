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
import com.davidbracewell.apollo.learning.classification.ClassifierResult;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.collection.Index;
import lombok.NonNull;
import org.apache.commons.math3.linear.RealVector;
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
  public ClassifierResult classify(@NonNull RealVector instance) {
    Counter<String> distribution = Counters.newHashMapCounter();
    for (int i = 0; i < numberOfLabels(); i++) {
      String label = getLabels().get(i);
      distribution.set(label, FastMath.log(priors[i]));
      for (int f = 0; f < numberOfFeatures(); f++) {
        if (instance.getEntry(f) != 0) {
          distribution.increment(label, FastMath.log(conditionals[f][i]));
        } else {
          distribution.increment(label, FastMath.log(1 - conditionals[f][i]));
        }
      }
      distribution.set(label, Math.exp(distribution.get(label)));
    }
    return new ClassifierResult(distribution);
  }


}//END OF BernoulliNaiveBayes

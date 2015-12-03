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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class BernoulliNaiveBayes extends NaiveBayes {
  private static final long serialVersionUID = 1L;

  /**
   * Instantiates a new Classifier.
   *
   * @param featureEncoder the feature encoder
   */
  protected BernoulliNaiveBayes(IndexEncoder labelEncoder, Encoder featureEncoder) {
    super(labelEncoder, featureEncoder);
  }


  @Override
  public ClassifierResult classify(@NonNull Vector instance) {
    Counter<String> distribution = Counters.newHashMapCounter();
    for (int i = 0; i < numberOfLabels(); i++) {
      String label = labelEncoder().decode(i).toString();
      distribution.set(label, FastMath.log(priors[i]));
      for (int f = 0; f < numberOfFeatures(); f++) {
        if (instance.get(f) != 0) {
          distribution.increment(label, FastMath.log(conditionals[f][i]));
        } else {
          distribution.increment(label, FastMath.log(1 - conditionals[f][i]));
        }
      }
      distribution.set(label, Math.exp(distribution.get(label)));
    }
    distribution.divideBySum();
    return new ClassifierResult(distribution);
  }


}//END OF BernoulliNaiveBayes

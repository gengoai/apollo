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
import com.davidbracewell.apollo.learning.IndexFeatureEncoder;
import com.davidbracewell.apollo.learning.classification.Classifier;
import com.davidbracewell.apollo.learning.classification.ClassifierResult;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.collection.Index;
import com.davidbracewell.stream.Streams;
import com.google.common.collect.Lists;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class BernoulliNaiveBayes extends NaiveBayes {
  private static final long serialVersionUID = 1L;

  /**
   * Instantiates a new Classifier.
   *
   * @param classLabels    the class labels
   * @param featureEncoder the feature encoder
   */
  protected BernoulliNaiveBayes(Index<String> classLabels, FeatureEncoder featureEncoder) {
    super(classLabels, featureEncoder);
  }


  @Override
  public ClassifierResult classify(@NonNull Vector instance) {
    Counter<String> distribution = Counters.newHashMapCounter();
    for (int i = 0; i < numberOfLabels(); i++) {
      String label = getLabels().get(i);
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


  public static void main(String[] args) throws Exception {
    Featurizer<String> featurizer = Featurizer
      .binary(str -> str.chars().mapToObj(i -> new String(new char[]{(char) i})).collect(Collectors.toSet()));
    BernoulliNaiveBayesLearner<String> learner = new BernoulliNaiveBayesLearner<>(IndexFeatureEncoder::new);

    List<String> data = Lists.newArrayList("ABBAD", "BAA", "A", "C", "AE", "B", "BE");

    Classifier classifier = learner
      .train(() -> Streams.of(data, false).map(str -> featurizer.extract(str, str.substring(0, 1))));


    System.out.println(classifier.getFeatureEncoder().features());
    System.out.println(classifier.getLabels());

    data.forEach(datum -> System.out.println(datum + " => " + classifier.classify(featurizer.extract(datum))));


  }


}//END OF BernoulliNaiveBayes

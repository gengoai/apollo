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
 *
 */

package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.stream.MStream;
import com.gengoai.stream.MDoubleAccumulator;

import java.util.Optional;

/**
 * <p>Transform values using <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">Tf-idf</a> </p>
 *
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;
   private volatile Counter<String> documentFrequencies = Counters.newCounter();
   private volatile double totalDocs = 0;


   /**
    * Instantiates a new Tfidf transform.
    */
   public TFIDFTransform() {
      super(null);
   }


   /**
    * Instantiates a new Tfidf transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public TFIDFTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public Instance applyInstance(Instance example) {
      final double sum = example.getFeatures()
                                .stream()
                                .filter(this::requiresProcessing)
                                .mapToDouble(Feature::getValue)
                                .sum();
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         double tfidf = in.getValue() / sum * Math.log(totalDocs / documentFrequencies.get(in.getName()));
         return Optional.of(Feature.realFeature(in.getName(), tfidf));
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      throw new UnsupportedOperationException();
   }

   @Override
   protected void fitInstances(MStream<Example> exampleStream) {
      MDoubleAccumulator docCount = exampleStream.getContext().doubleAccumulator(0d);
      this.documentFrequencies.merge(exampleStream.flatMap(e -> {
         docCount.add(1.0);
         return e.getFeatureNameSpace().distinct();
      }).countByValue());
      this.totalDocs = docCount.value();
   }

   @Override
   public void reset() {
      totalDocs = 0;
      documentFrequencies.clear();
   }

   @Override
   public String toString() {
      return "TFIDFTransform[" + getRestriction() + "]{totalDocuments=" + totalDocs +
                ", vocabSize=" + documentFrequencies.size() + "}";
   }


}//END OF TFIDFTransform

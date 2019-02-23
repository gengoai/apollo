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

package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;

import java.util.List;
import java.util.stream.Stream;

/**
 * The type Count feature vectorizer.
 *
 * @author David B. Bracewell
 */
public class CountFeatureVectorizer extends IndexVectorizer {
   private static final long serialVersionUID = 1L;

   public CountFeatureVectorizer() {
   }

   public CountFeatureVectorizer(String unknown) {
      super(unknown);
   }

   public CountFeatureVectorizer(List<String> alphabet, String unknown) {
      super(alphabet, unknown);
   }

   @Override
   protected Stream<String> getAlphabetSpace(Example example) {
      return example.getFeatureNameSpace();
   }

   @Override
   public NDArray transform(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(size());
      for (Feature feature : example.getFeatures()) {
         int fi = indexOf(feature.getName());
         if (fi >= 0) {
            ndArray.increment(fi, feature.getValue());
         }
      }
      return ndArray;
   }

}//END OF CountFeatureVectorizer

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
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Index;

import java.util.Collections;
import java.util.Set;

/**
 * Uses the hashing trick to reduce the feature space by hashing feature names into a given number of buckets.
 *
 * @author David B. Bracewell
 */
public class HashingEncoder implements DiscreteVectorizer {
   private static final long serialVersionUID = 1L;
   private final boolean isBinary;
   private final int numberOfFeatures;

   /**
    * Instantiates a new HashingEncoder.
    *
    * @param numberOfFeatures the number of features to represent
    * @param isBinary         True - treat all features binary, False treat them as accumulated real values.
    */
   public HashingEncoder(int numberOfFeatures, boolean isBinary) {
      this.numberOfFeatures = numberOfFeatures;
      this.isBinary = isBinary;
   }

   @Override
   public Set<String> alphabet() {
      return Collections.emptySet();
   }

   @Override
   public Index<String> asIndex() {
      throw new UnsupportedOperationException();
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public String getString(double value) {
      throw new UnsupportedOperationException();
   }

   @Override
   public int indexOf(String value) {
      return (value.hashCode() & 0x7fffffff) % numberOfFeatures;
   }

   @Override
   public int size() {
      return numberOfFeatures;
   }

   @Override
   public NDArray transform(Example example) {
      NDArray ndArray = NDArrayFactory.ND.array(size());
      for (Feature feature : example.getFeatures()) {
         int fi = indexOf(feature.getName());
         if (isBinary) {
            ndArray.set(fi, 1.0);
         } else {
            ndArray.set(fi, ndArray.get(fi) + feature.getValue());
         }
      }
      return ndArray;
   }

}//END OF HashingEncoder

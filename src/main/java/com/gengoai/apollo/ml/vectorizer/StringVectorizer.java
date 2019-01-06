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

/**
 * @author David B. Bracewell
 */
public abstract class StringVectorizer implements Vectorizer<String> {
   private final boolean isLabelVectorizer;

   protected StringVectorizer(boolean isLabelVectorizer) {
      this.isLabelVectorizer = isLabelVectorizer;
   }

   private void setIf(NDArray ndArray, String name, double value) {
      int index = (int) encode(name);
      if (index >= 0) {
         ndArray.set(index, value);
      }
   }

   private void setIf(NDArray ndArray, int row, String name, double value) {
      int index = (int) encode(name);
      if (index >= 0) {
         ndArray.set(row, index, value);
      }
   }

   @Override
   public NDArray transform(Example example) {
      if (example.isInstance()) {
         return transformInstance(example);
      }
      return transformSequence(example);
   }

   protected NDArray transformInstance(Example example) {
      final NDArray ndArray = NDArrayFactory.DEFAULT().zeros(size());
      if (isLabelVectorizer) {
         example.getLabelAsSet().forEach(label -> setIf(ndArray, label, 1.0));
      } else {
         example.getFeatures().forEach(feature -> setIf(ndArray, feature.name, feature.value));
      }
      return ndArray;
   }

   protected NDArray transformSequence(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(example.size(), size());
      for (int row = 0; row < example.size(); row++) {
         Example child = example.getExample(row);
         final int childIndex = row;
         if (isLabelVectorizer) {
            child.getLabelAsSet().forEach(label -> setIf(ndArray, childIndex, label, 1.0));
         } else {
            child.getFeatures().forEach(feature -> setIf(ndArray, childIndex, feature.name, feature.value));
         }
      }
      return ndArray;
   }



}//END OF StringVectorizer

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

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.json.JsonEntry;

import java.util.Collections;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class HashingEncoder extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final boolean isBinary;
   private final int numberOfFeatures;

   public HashingEncoder(int numberOfFeatures, boolean isBinary) {
      super(false);
      this.numberOfFeatures = numberOfFeatures;
      this.isBinary = isBinary;
   }

   @Override
   public Set<String> alphabet() {
      return Collections.emptySet();
   }

   @Override
   public String decode(double value) {
      return null;
   }

   @Override
   public double encode(String value) {
      return (value.hashCode() & 0x7fffffff) % numberOfFeatures;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return numberOfFeatures;
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", HashingEncoder.class)
                      .addProperty("numberOfFeatures", numberOfFeatures)
                      .addProperty("isBinary", isBinary);
   }

   @Override
   public String toString() {
      return "HashingEncoder{" +
                "numberOfFeatures=" + numberOfFeatures +
                ", isBinary=" + isBinary +
                '}';
   }
}//END OF HashingEncoder

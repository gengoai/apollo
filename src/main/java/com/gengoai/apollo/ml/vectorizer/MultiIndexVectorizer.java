/*
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

package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.collection.Index;
import com.gengoai.collection.Indexes;
import com.gengoai.tuple.Tuple2;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static com.gengoai.tuple.Tuples.$;

public class MultiIndexVectorizer implements DiscreteVectorizer {
   public static final String UNKNOWN = "---UNKNOWN---";
   private final Set<String> combinedIndex = new HashSet<>();
   private final Map<String, Index<String>> prefixIndexes = new ConcurrentHashMap<>();
   private final List<String> combinedOrder = new ArrayList<>();


   @Override
   public Set<String> alphabet() {
      return combinedIndex;
   }

   @Override
   public Index<String> asIndex() {
      throw new UnsupportedOperationException();
   }

   @Override
   public String getString(double value) {
      throw new UnsupportedOperationException();
   }

   @Override
   public int indexOf(String value) {
      final String prefix = Feature.getPrefix(value);
      Index<String> index = prefixIndexes.get(prefix);
      if(index == null) {
         throw new IllegalArgumentException(String.format("Invalid prefix '%s'", prefix));
      }
      int id = index.getId(Feature.getSuffix(value));
      return Math.max(id, 0);
   }

   @Override
   public int size() {
      throw new UnsupportedOperationException();
   }

   @Override
   public void fit(ExampleDataset dataset) {
      combinedIndex.clear();
      prefixIndexes.clear();
      combinedOrder.clear();
      dataset.stream()
            .flatMap(Example::getFeatureNameSpace)
            .forEach(f -> {
               final String prefix = Feature.getPrefix(f);
               final String suffix = Feature.getSuffix(f);
               prefixIndexes.computeIfAbsent(prefix, k -> Indexes.indexOf(UNKNOWN)).add(suffix);
               combinedIndex.add(f);
            });
      combinedOrder.addAll(new TreeSet<>(prefixIndexes.keySet()));
   }

   @Override
   public NDArray transform(Example example) {
      Map<String, Integer> m = example.getFeatureNameSpace()
            .map(s -> $(Feature.getPrefix(s), indexOf(s)))
            .collect(Collectors.toMap(Tuple2::getKey, Tuple2::getV2));
      NDArray ndArray = NDArrayFactory.ND.array(prefixIndexes.size());
      int i = 0;
      for(String s : combinedOrder) {
         ndArray.set(i, m.get(s));
         i++;
      }
      return ndArray;
   }
}//END OF MultiIndexVectorizer

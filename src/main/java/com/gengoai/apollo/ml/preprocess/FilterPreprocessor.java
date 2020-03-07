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

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.ExampleDataset;

import java.io.Serializable;
import java.util.Collection;
import java.util.Set;
import java.util.stream.Collectors;

import static com.gengoai.collection.Sets.asHashSet;
import static com.gengoai.collection.Sets.hashSetOf;

/**
 * <p>A preprocessor that removes all features of a given prefix</p>
 *
 * @author David B. Bracewell
 */
public class FilterPreprocessor implements InstancePreprocessor, Serializable {
   private static final long serialVersionUID = 1L;
   private final Set<String> toRemove;

   /**
    * Instantiates a new Filter preprocessor.
    *
    * @param toRemove the to remove
    */
   public FilterPreprocessor(String... toRemove) {
      this.toRemove = hashSetOf(toRemove);
   }

   /**
    * Instantiates a new Filter preprocessor.
    *
    * @param toRemove the to remove
    */
   public FilterPreprocessor(Collection<String> toRemove) {
      this.toRemove = asHashSet(toRemove);
   }

   @Override
   public Instance applyInstance(Instance example) {
      Instance ii = new Instance(example.getLabel(), example.getFeatures().stream()
                                                            .filter(f -> !toRemove.contains(f.getPrefix()))
                                                            .collect(Collectors.toList()));
      ii.setWeight(example.getWeight());
      return ii;
   }

   @Override
   public ExampleDataset fitAndTransform(ExampleDataset dataset) {
      return dataset.map(this::apply);
   }

   @Override
   public void reset() {

   }
}//END OF FilterPreprocessor

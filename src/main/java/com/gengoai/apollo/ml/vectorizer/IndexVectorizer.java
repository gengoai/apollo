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

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.collection.HashMapIndex;
import com.gengoai.collection.Index;
import com.gengoai.collection.Sets;
import com.gengoai.stream.Streams;

import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Base class for indexing vectorizers.
 *
 * @author David B. Bracewell
 */
public abstract class IndexVectorizer implements DiscreteVectorizer {
   private static final long serialVersionUID = 1L;
   /**
    * The Fixed.
    */
   protected final boolean fixed;
   /**
    * The Index.
    */
   protected final Index<String> index = new HashMapIndex<>();
   /**
    * The Unknown.
    */
   protected final String unknown;


   /**
    * Instantiates a new IndexVectorizer.
    */
   public IndexVectorizer() {
      this(null);
   }

   /**
    * Instantiates a new IndexVectorizer.
    *
    * @param unknown the unknown feature/label
    */
   public IndexVectorizer(String unknown) {
      this.unknown = unknown;
      if (this.unknown != null) {
         index.add(this.unknown);
      }
      this.fixed = false;
   }

   /**
    * Instantiates a new IndexVectorizer with a fixed alphabet.
    *
    * @param alphabet the fixed alphabet
    * @param unknown  the unknown feature/label
    */
   public IndexVectorizer(List<String> alphabet, String unknown) {
      this.unknown = unknown;
      index.addAll(alphabet);
      this.fixed = true;
   }

   @Override
   public Set<String> alphabet() {
      return Sets.asHashSet(index);
   }

   @Override
   public Index<String> asIndex() {
      return index;
   }

   @Override
   public void fit(ExampleDataset dataset) {
      if (!fixed) {
         index.clear();
         index.addAll(dataset.stream()
                             .flatMap(Streams::asStream)
                             .flatMap(this::getAlphabetSpace)
                             .distinct()
                             .collect());
      }
   }

   /**
    * Gets the strings for the alphabet space from an example
    *
    * @param example the example
    * @return the alphabet space
    */
   protected abstract Stream<String> getAlphabetSpace(Example example);

   @Override
   public String getString(double value) {
      return index.get((int) value);
   }

   @Override
   public String unknown() {
      return unknown;
   }

   @Override
   public int indexOf(String value) {
      int i = index.getId(value);
      if (i < 0 && unknown != null) {
         return index.getId(unknown);
      }
      return i;
   }

   @Override
   public int size() {
      return index.size();
   }


}//END OF IndexEncoder

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

import cc.mallet.types.Alphabet;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.collection.Index;
import com.gengoai.collection.Sets;
import com.gengoai.conversion.Cast;

import java.util.Arrays;
import java.util.Set;

/**
 * Specialized Vectorizer for Mallet models
 *
 * @author David B. Bracewell
 */
public class MalletVectorizer implements DiscreteVectorizer {
   private static final long serialVersionUID = 1L;
   private Alphabet alphabet;

   /**
    * Instantiates a new MalletVectorizer.
    *
    * @param alphabet the alphabet
    */
   public MalletVectorizer(Alphabet alphabet) {
      this.alphabet = alphabet;
   }

   @Override
   public Set<String> alphabet() {
      return Cast.cast(Sets.asHashSet(Arrays.asList(alphabet.toArray())));
   }

   @Override
   public Index<String> asIndex() {
      throw new UnsupportedOperationException();
   }

   @Override
   public void fit(ExampleDataset dataset) {

   }

   /**
    * Gets the underlying Mallet alphabet.
    *
    * @return the alphabet
    */
   public Alphabet getAlphabet() {
      return alphabet;
   }

   /**
    * Sets the underlying MAllet Alphabet.
    *
    * @param alphabet the alphabet
    */
   public void setAlphabet(Alphabet alphabet) {
      this.alphabet = alphabet;
   }

   @Override
   public String getString(double value) {
      return alphabet.lookupObject((int) value).toString();
   }

   @Override
   public int indexOf(String value) {
      return alphabet.lookupIndex(value);
   }

   @Override
   public int size() {
      return alphabet.size();
   }

   @Override
   public NDArray transform(Example example) {
      return null;
   }

}// END OF MalletFeatureEncoder

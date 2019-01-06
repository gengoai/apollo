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
import com.gengoai.collection.Index;
import com.gengoai.collection.Indexes;
import com.gengoai.collection.Sets;

import java.util.Collection;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class FixedAlphabetVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   final Index<String> alphabet;

   public FixedAlphabetVectorizer(boolean isLabelVectorizer, Collection<String> alphabet) {
      super(isLabelVectorizer);
      this.alphabet = Indexes.indexOf(alphabet);
   }

   @Override
   public Set<String> alphabet() {
      return Sets.asHashSet(alphabet);
   }

   @Override
   public String decode(double value) {
      return alphabet.get((int) value);
   }

   @Override
   public double encode(String value) {
      return alphabet.getId(value);
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return alphabet.size();
   }
}//END OF FixedAlphabetVectorizer

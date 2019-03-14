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
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Index;
import com.gengoai.collection.Indexes;

import java.util.Collections;
import java.util.Set;

/**
 * The type No opt vectorizer.
 *
 * @author David B. Bracewell
 */
public class NoOptVectorizer implements DiscreteVectorizer {
   private static final long serialVersionUID = 1L;
   /**
    * The constant INSTANCE.
    */
   public static final DiscreteVectorizer INSTANCE = new NoOptVectorizer();


   private NoOptVectorizer() {

   }

   @Override
   public Set<String> alphabet() {
      return Collections.emptySet();
   }

   @Override
   public Index<String> asIndex() {
      return Indexes.indexOf();
   }

   @Override
   public String getString(double value) {
      throw null;
   }

   @Override
   public int indexOf(String value) {
      return -1;
   }

   @Override
   public int size() {
      return 0;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public NDArray transform(Example example) {
      throw new UnsupportedOperationException();
   }
}//END OF NoOptVectorizer

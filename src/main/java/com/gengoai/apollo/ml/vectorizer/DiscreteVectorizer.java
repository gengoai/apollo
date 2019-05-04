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

import com.gengoai.Copyable;
import com.gengoai.collection.Index;

import java.util.Set;

/**
 * <p>A vectorizer to handle transform one or more discrete values into an {@link com.gengoai.apollo.linear.NDArray}</p>
 *
 * @author David B. Bracewell
 */
public interface DiscreteVectorizer extends Vectorizer, Copyable<DiscreteVectorizer> {

   /**
    * The discrete set of strings representing the alphabet
    *
    * @return the alphabet
    */
   Set<String> alphabet();

   /**
    * Provides an Index view of the alphabet.
    *
    * @return the index view over the alphabet
    */
   Index<String> asIndex();

   @Override
   default DiscreteVectorizer copy() {
      return Copyable.deepCopy(this);
   }

   /**
    * Decodes the given index returning the associated string feature.
    *
    * @param value the value
    * @return the String feature or null if the value is invalid
    */
   String getString(double value);

   /**
    * Encodes a feature name into an index
    *
    * @param value the feature name
    * @return the encoded index
    */
   int indexOf(String value);

   /**
    * The unique number of features or labels known by the vectorizer.
    *
    * @return the number of features / labels in the vectorizer.
    */
   int size();

   default String unknown() {
      return null;
   }

}//END OF DiscreteVectorizer

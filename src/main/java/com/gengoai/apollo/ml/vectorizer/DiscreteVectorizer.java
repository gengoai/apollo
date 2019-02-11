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

import java.util.Set;

/**
 * <p>A vectorizer to handle transform one or more discrete values into an {@link com.gengoai.apollo.linear.NDArray}</p>
 *
 * @author David B. Bracewell
 */
public interface DiscreteVectorizer extends Vectorizer {

   /**
    * Alphabet set.
    *
    * @return the set
    */
   Set<String> alphabet();

   /**
    * Decode t.
    *
    * @param value the value
    * @return the t
    */
   String getString(double value);

   /**
    * Encode double.
    *
    * @param value the value
    * @return the double
    */
   int indexOf(String value);


   /**
    * The unique number of features or labels known by the vectorizer.
    *
    * @return the number of features / labels in the vectorizer.
    */
   int size();

}//END OF DiscreteVectorizer

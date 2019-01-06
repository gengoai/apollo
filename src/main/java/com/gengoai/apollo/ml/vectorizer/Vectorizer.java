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

import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

/**
 * <p> Encoders are responsible for encoding and decoding objects into double values. Encoders are used to create
 * vector representations of features. </p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Vectorizer<T> extends Serializable {

   default Set<String> alphabet() {
      return Collections.emptySet();
   }

   /**
    * Decode t.
    *
    * @param value the value
    * @return the t
    */
   T decode(double value);

   /**
    * Encode double.
    *
    * @param value the value
    * @return the double
    */
   double encode(T value);

   /**
    * Fit.
    *
    * @param dataset the dataset
    */
   void fit(Dataset dataset);

   /**
    * Size int.
    *
    * @return the int
    */
   int size();

   /**
    * Transform nd array.
    *
    * @param example the example
    * @return the nd array
    */
   NDArray transform(Example example);


}// END OF Encoder

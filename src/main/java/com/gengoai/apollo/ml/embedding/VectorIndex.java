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

package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.NDArray;

import java.util.stream.Stream;

/**
 * <p>An integer indexable array of NDArray.</p>
 *
 * @author David B. Bracewell
 */
public interface VectorIndex {

   /**
    * Retrieves the NDArray for the given index
    *
    * @param index the index of the NDArray
    * @return the NDArray of the given index
    */
   NDArray lookup(int index);

   /**
    * Stream of NDArray in in the index
    *
    * @return the stream
    */
   Stream<NDArray> stream();

   /**
    * the dimension of the vectors in the index
    *
    * @return the dimension
    */
   int dimension();


   /**
    * the number of vectors in the index
    *
    * @return the number of vectors in the index
    */
   int size();

}//END OF VectorIndex

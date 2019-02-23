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

import java.io.Serializable;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class DefaultVectorIndex implements VectorIndex, Serializable {
   private static final long serialVersionUID = 1L;
   private final NDArray[] vectors;

   public DefaultVectorIndex(NDArray[] vectors) {
      this.vectors = vectors;
   }

   @Override
   public NDArray lookup(int index) {
      return vectors[index].copy();
   }

   @Override
   public Stream<NDArray> stream() {
      return Stream.of(vectors);
   }

   @Override
   public int dimension() {
      return (int) vectors[0].length();
   }
}//END OF DefaultVectorIndex

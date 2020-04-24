/*
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
 */

package com.gengoai.apollo.ml.encoder;

import com.gengoai.apollo.math.linalg.NDArray;
import com.gengoai.apollo.math.linalg.NDArrayFactory;
import com.gengoai.apollo.ml.DataSet;
import com.gengoai.apollo.ml.observation.Observation;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.apollo.ml.observation.VariableNameSpace;
import lombok.NonNull;

import java.util.Collection;

/**
 * <p>Commonly used methods when working {@link Encoder}s</p>
 *
 * @author David B. Bracewell
 */
public final class EncodeUtils {

   /**
    * Fits the given {@link Encoder} to the given {@link DataSet} for the given sources.
    *
    * @param encoder   the encoder
    * @param dataset   the dataset
    * @param sources   the sources
    * @param nameSpace the {@link VariableNameSpace} to use when fitting the dataset
    */
   public static void fit(@NonNull Encoder encoder,
                          @NonNull DataSet dataset,
                          @NonNull Collection<String> sources,
                          @NonNull VariableNameSpace nameSpace) {
      encoder.fit(dataset.stream()
                         .flatMap(d -> d.stream(sources))
                         .flatMap(Observation::getVariableSpace)
                         .map(nameSpace::transform));
   }

   /**
    * Encodes the variable space of the given {@link Observation} into an {@link NDArray} using the given encoder where
    * the value of an index in the array is <code>1</code> when that {@link Variable} occurred at least once. The
    * resulting array has a shape of <code>1 x Encoder.size()</code>
    *
    * @param observation the observation
    * @param encoder     the encoder
    * @param nameSpace   the {@link VariableNameSpace} to use when encoding.
    * @return the NDArray
    */
   public static NDArray toBinaryVector(@NonNull Observation observation,
                                        @NonNull Encoder encoder,
                                        @NonNull VariableNameSpace nameSpace) {
      NDArray n = NDArrayFactory.ND.array(encoder.size());
      observation.getVariableSpace()
                 .forEach(v -> {
                    int index = encoder.encode(nameSpace.getName(v));
                    if(index >= 0) {
                       n.set(index, 1);
                    }
                 });
      return n;
   }

   /**
    * Encodes the variable space of the given {@link Observation} into an {@link NDArray} using the given encoder where
    * the value an index in the array is equal to the sum of the values of all occurrences of the associated {@link
    * Variable}. The resulting array has a shape of <code>1 x Encoder.size()</code>
    *
    * @param observation the observation
    * @param encoder     the encoder
    * @param nameSpace   the {@link VariableNameSpace} to use when encoding.
    * @return the NDArray
    */
   public static NDArray toCountVector(@NonNull Observation observation,
                                       @NonNull Encoder encoder,
                                       @NonNull VariableNameSpace nameSpace) {
      NDArray n = NDArrayFactory.ND.array(encoder.size());
      observation.getVariableSpace()
                 .forEach(v -> {
                    int index = encoder.encode(nameSpace.getName(v));
                    if(index >= 0) {
                       n.set(index, n.get(index) + v.getValue());
                    }
                 });
      return n;
   }

   private EncodeUtils() {
      throw new IllegalAccessError();
   }

}//END OF EncodeUtils

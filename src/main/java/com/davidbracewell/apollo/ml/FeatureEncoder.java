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
 */

package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.DynamicSparseVector;
import com.davidbracewell.apollo.linalg.NamedVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.NonNull;

import java.util.Collection;
import java.util.List;

/**
 * The interface Feature encoder.
 *
 * @author David B. Bracewell
 */
public interface FeatureEncoder {

  /**
   * Decodes the given value.
   *
   * @param value the encoded value
   * @return the decoded value
   */
  String decode(int value);

  /**
   * Encodes a given feature into a double.
   *
   * @param feature the feature to encode
   * @return the encoded value
   */
  int encode(@NonNull String feature);

  /**
   * The decoded feature values.
   *
   * @return the collection of decoded feature values.
   */
  Collection<String> features();

  /**
   * Freezes the encoder. When frozen no new values will be added to the encoder.
   */
  void freeze();

  /**
   * Unfreezes the encoder allowing new values to be added.
   */
  void unFreeze();

  /**
   * Checks if the encoder allows adding new values for index-like encoders.
   *
   * @return True - Frozen, False not Frozen
   */
  boolean isFrozen();

  /**
   * The number of encoded values
   *
   * @return the number of encoded values.
   */
  int size();

  /**
   * To vector vector.
   *
   * @param instance the instance
   * @return the vector
   */
  default Vector toVector(@NonNull Instance instance) {
    DynamicSparseVector vector = new DynamicSparseVector(this::size);
    instance.forEach(feature -> {
      int index = encode(feature.getName());
      if (index != -1) {
        vector.set(index, feature.getValue());
      }
    });
    return vector;
  }

  /**
   * To vectors list.
   *
   * @param instances the instances
   * @return the list
   */
  default List<NamedVector> toVectors(@NonNull Collection<Instance> instances) {
    return toVectors(Streams.of(instances, false));
  }

  /**
   * To vectors list.
   *
   * @param instances the instances
   * @return the list
   */
  default List<NamedVector> toVectors(@NonNull MStream<Instance> instances) {
    return instances
      .map(instance -> new NamedVector(this.toVector(instance), instance.getLabel().toString()))
      .collect();
  }


}//END OF FeatureEncoder

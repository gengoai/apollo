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

package com.davidbracewell.apollo.learning;

import com.davidbracewell.collection.Counter;
import com.davidbracewell.function.SerializableFunction;
import lombok.NonNull;

import java.util.*;

/**
 * The interface Featurizer.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public interface Featurizer<T> extends SerializableFunction<T, Set<Feature>> {

  /**
   * Binary featurizer.
   *
   * @param <T>      the type parameter
   * @param function the function
   * @return the featurizer
   */
  static <T> Featurizer<T> binary(@NonNull SerializableFunction<? super T, ? extends Set<String>> function) {
    return new BinaryFeaturizer<T>() {
      private static final long serialVersionUID = 1L;

      @Override
      protected Set<String> extract(T t) {
        return function.apply(t);
      }
    };
  }

  /**
   * Real featurizer.
   *
   * @param <T>      the type parameter
   * @param function the function
   * @return the featurizer
   */
  static <T> Featurizer<T> real(@NonNull SerializableFunction<? super T, ? extends Counter<String>> function) {
    return new RealFeaturizer<T>() {
      private static final long serialVersionUID = 1L;

      @Override
      protected Counter<String> extract(T t) {
        return function.apply(t);
      }
    };
  }


  /**
   * Builder builder.
   *
   * @param <T> the type parameter
   * @return the builder
   */
  static <T> Builder<T> builder() {
    return new Builder<>();
  }

  /**
   * Of featurizer.
   *
   * @param <T>         the type parameter
   * @param featurizers the featurizers
   * @return the featurizer
   */
  @SafeVarargs
  static <T> Featurizer<T> of(@NonNull Featurizer<? super T>... featurizers) {
    return Featurizer.<T>builder().addAll(Arrays.asList(featurizers)).build();
  }


  /**
   * The type Builder.
   *
   * @param <T> the type parameter
   */
  class Builder<T> {
    private final Set<Featurizer<? super T>> featurizers = new LinkedHashSet<>();

    /**
     * Add builder.
     *
     * @param featurizer the featurizer
     * @return the builder
     */
    public Builder<T> add(@NonNull Featurizer<? super T> featurizer) {
      featurizers.add(featurizer);
      return this;
    }

    /**
     * Add all builder.
     *
     * @param featurizers the featurizers
     * @return the builder
     */
    public Builder<T> addAll(@NonNull Collection<? extends Featurizer<? super T>> featurizers) {
      featurizers.forEach(this.featurizers::add);
      return this;
    }

    /**
     * Build featurizer.
     *
     * @return the featurizer
     */
    public Featurizer<T> build() {
      return new Featurizer<T>() {
        private static final long serialVersionUID = 1L;
        private final Set<Featurizer<? super T>> featurizers = new LinkedHashSet<>(Builder.this.featurizers);

        @Override
        public Set<Feature> apply(T t) {
          Set<Feature> features = new HashSet<>();
          featurizers.forEach(f -> features.addAll(f.apply(t)));
          return features;
        }
      };
    }


  }


}//END OF Featurizer

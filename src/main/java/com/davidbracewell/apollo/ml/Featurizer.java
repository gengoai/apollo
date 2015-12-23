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

import com.davidbracewell.collection.Counter;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * <p> A featurizer converts an input into a one or more <code>Feature</code>s which have a name and a value. Specific
 * implementations may implement the generic featurizer interface or specialize the Binary or Real featurizers. </p>
 *
 * @param <INPUT> the type of the input being converted into features
 * @author David B. Bracewell
 */
public interface Featurizer<INPUT> extends SerializableFunction<INPUT, Set<Feature>> {

  /**
   * Creates a binary featurizer, i.e. one that in which the values returned are all 1.0 (true). The given function
   * converts the input into a collection of feature names which are assumed to have the value true (1.0).
   *
   * @param <T>      the type of the input
   * @param function the function to use to convert the input int feature names.
   * @return the featurizer
   */
  static <T> Featurizer<T> binary(@NonNull SerializableFunction<? super T, ? extends Collection<String>> function) {
    return new BinaryFeaturizer<T>() {
      private static final long serialVersionUID = 1L;

      @Override
      protected Set<String> applyImpl(T t) {
        Collection<String> c = function.apply(t);
        if (c instanceof Set) {
          return Cast.as(c);
        }
        return new HashSet<>(c);
      }
    };
  }

  /**
   * Creates a real featurizer that uses a given function that converts the input into a counter of features.
   *
   * @param <T>      the type of the input
   * @param function the function to use to convert the input
   * @return the featurizer
   */
  static <T> Featurizer<T> real(@NonNull SerializableFunction<? super T, ? extends Counter<String>> function) {
    return new RealFeaturizer<T>() {
      private static final long serialVersionUID = 1L;

      @Override
      protected Counter<String> applyImpl(T t) {
        return function.apply(t);
      }
    };
  }

  /**
   * Creates a new builder that allows constructing more complex sets of featurizers.
   *
   * @param <T> the type of the input
   * @return the builder
   */
  static <T> Builder<T> builder() {
    return new Builder<>();
  }

  /**
   * Converts the given input into features and creates an <code>Instance</code> from the features.
   *
   * @param object the input
   * @return the instance
   */
  default Instance extract(@NonNull INPUT object) {
    return Instance.create(apply(object));
  }

  /**
   * Converts the given input into features and creates an <code>Instance</code> from the features.
   *
   * @param object the input
   * @param label  the label to assign the input
   * @return the instance
   */
  default Instance extract(@NonNull INPUT object, Object label) {
    return Instance.create(apply(object), label);
  }

  /**
   * Extract instance.
   *
   * @param labeledDatum the labeled datum
   * @return the instance
   */
  default Instance extract(@NonNull LabeledDatum<INPUT> labeledDatum) {
    return Instance.create(apply(labeledDatum.getData()), labeledDatum.getLabel());
  }


  /**
   * A builder that allows combining multiple featurizers
   *
   * @param <T> the type of the input
   */
  class Builder<T> {
    private final Set<Featurizer<? super T>> featurizers = new LinkedHashSet<>();

    /**
     * Adds the featurizer to the builder
     *
     * @param featurizer the featurizer to add
     * @return the builder
     */
    public Builder<T> add(Featurizer<? super T> featurizer) {
      if (featurizer != null) {
        featurizers.add(featurizer);
      }
      return this;
    }

    /**
     * Adds all featurizers in the given collection.
     *
     * @param featurizers the featurizers to add
     * @return the builder
     */
    public Builder<T> addAll(Collection<? extends Featurizer<? super T>> featurizers) {
      if (featurizers != null) {
        featurizers.forEach(this.featurizers::add);
      }
      return this;
    }

    /**
     * Builds a featurizer that chains all the add featurizers into one.
     *
     * @return the featurizer
     */
    public Featurizer<T> build() {
      Preconditions.checkState(featurizers.size() > 0, "No Featurizers have been added.");
      if (featurizers.size() == 1) {
        return Cast.as(featurizers.stream().findFirst().get());
      }
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

  }//END OF Builder


}//END OF Featurizer

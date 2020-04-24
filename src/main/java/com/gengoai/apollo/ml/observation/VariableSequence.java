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

package com.gengoai.apollo.ml.observation;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * <p>
 * A {@link Sequence} where the observations are {@link Variable}s. This implementation works well for raw sequence
 * input or sequence label output, but care must * be taken when filtering Variables as null values will replace the
 * filtered values.
 * </p>
 *
 * @author David B. Bracewell
 */
public class VariableSequence extends ArrayList<Variable> implements Sequence<Variable> {

   /**
    * Instantiates a new VariableSequence
    */
   public VariableSequence() {
   }

   /**
    * Instantiates a new VariableSequence.
    *
    * @param collection the Collection of variables
    */
   public VariableSequence(@NonNull Collection<? extends Variable> collection) {
      super(collection);
   }

   /**
    * Instantiates a new VariableSequence.
    *
    * @param stream the stream of variables
    */
   public VariableSequence(@NonNull Stream<? extends Variable> stream) {
      stream.forEach(this::add);
   }

   @Override
   public VariableSequence copy() {
      return new VariableSequence(stream().map(Variable::copy));
   }

   @Override
   public void mapVariables(@NonNull Function<Variable, Variable> mapper) {
      for(int i = 0; i < size(); i++) {
         set(i, mapper.apply(get(i)));
      }
   }

   @Override
   public void removeVariables(@NonNull Predicate<Variable> filter) {
      for(int i = 0; i < size(); i++) {
         if(filter.test(get(i))) {
            set(i, null);
         }
      }
   }

   @Override
   public void updateVariables(@NonNull Consumer<Variable> updater) {
      forEach(updater);
   }

}//END OF VariableSequence

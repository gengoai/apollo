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

package com.gengoai.apollo.ml;

import com.gengoai.conversion.Cast;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.collection.Lists.arrayListOf;

/**
 * <p>A Sequence represents an example made up of a finite enumerated list of child examples ({@link Instance}s).
 * Sequences do not allow direct access to labels and features as the labels and features are associated with the
 * examples in the sequence.</p>
 *
 * @author David B. Bracewell
 */
public class Sequence extends Example {
   private static final long serialVersionUID = 1L;
   private final ArrayList<Instance> sequence = new ArrayList<>();


   /**
    * Instantiates a new empty Sequence.
    */
   public Sequence() {

   }

   /**
    * Creates an unlabeled sequence from a sequence of strings.
    *
    * @param tokens the tokens
    * @return the sequence
    */
   public static Sequence create(Stream<String> tokens) {
      Sequence seq = new Sequence();
      tokens.forEach(token -> seq.sequence.add(new Instance(null, arrayListOf(Feature.booleanFeature(token)))));
      return seq;
   }

   /**
    * Creates an unlabeled sequence from a sequence of strings.
    *
    * @param tokens the tokens
    * @return the sequence
    */
   public static Sequence create(List<String> tokens) {
      return create(tokens.stream());
   }

   /**
    * Instantiates a new Sequence with weight 1.0 with the given child examples.
    *
    * @param examples the child examples (i.e. sequence instances in order)
    */
   public Sequence(List<? extends Example> examples) {
      examples.forEach(this::add);
      this.sequence.trimToSize();
   }

   @Override
   public void add(Example example) {
      checkArgument(example instanceof Instance, "Can only add Instance examples.");
      sequence.add(Cast.as(example));
   }

   @Override
   public Example copy() {
      Sequence copy = new Sequence(this.sequence);
      copy.setWeight(getWeight());
      return copy;
   }

   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Sequence)) return false;
      Sequence examples = (Sequence) o;
      return Objects.equals(sequence, examples.sequence);
   }

   @Override
   public Example getExample(int index) {
      return sequence.get(index);
   }

   @Override
   public boolean hasLabel() {
      for (int i = 0; i < size(); i++) {
         if (getExample(i).hasLabel()) {
            return true;
         }
      }
      return false;
   }

   @Override
   public int hashCode() {
      return Objects.hash(sequence);
   }

   @Override
   public Sequence mapInstance(Function<Instance, Instance> mapper) {
      Sequence ii = new Sequence(sequence.stream()
                                         .map(mapper)
                                         .collect(Collectors.toList()));
      ii.setWeight(getWeight());
      return ii;
   }

   @Override
   public int size() {
      return sequence.size();
   }

   @Override
   public String toString() {
      return "Sequence{" +
                "sequence=" + sequence +
                ", weight=" + getWeight() +
                '}';
   }


}//END OF Sequence

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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * A sequence of {@link LabeledDatum} used to extract {@link Sequence} for classified sequence labeling.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class LabeledSequence<T> implements Serializable, Iterable<LabeledDatum<T>> {
   private static final long serialVersionUID = 1L;
   private List<LabeledDatum<T>> sequence = new ArrayList<>();


   /**
    * Instantiates a new empty Labeled sequence.
    */
   public LabeledSequence() {

   }

   /**
    * Instantiates a new Labeled sequence for the given data with no labels.
    *
    * @param data the data
    */
   public LabeledSequence(List<T> data) {
      for (T datum : data) {
         sequence.add(LabeledDatum.of(null, datum));
      }
   }

   /**
    * Instantiates a new Labeled sequence made up of the given instances..
    *
    * @param instances the instances
    */
   @SafeVarargs
   public LabeledSequence(LabeledDatum<T>... instances) {
      Collections.addAll(sequence, instances);
   }

   /**
    * Adds a labeled instance to the sequence.
    *
    * @param datum the datum to add
    */
   public void add(LabeledDatum<T> datum) {
      sequence.add(datum);
   }

   /**
    * Adds a labeled instance to the sequence.
    *
    * @param label the data label
    * @param data  the data
    */
   public void add(Object label, T data) {
      sequence.add(LabeledDatum.of(label, data));
   }

   /**
    * Gets the {@link LabeledDatum} at the given index.
    *
    * @param index the index
    * @return the labeled datum
    */
   public LabeledDatum<T> get(int index) {
      return sequence.get(index);
   }

   @Override
   public Iterator<LabeledDatum<T>> iterator() {
      return sequence.iterator();
   }

   /**
    * The number of items in the sequence.
    *
    * @return the number of items in the sequence.
    */
   public int size() {
      return sequence.size();
   }

}//END OF LabeledSequence

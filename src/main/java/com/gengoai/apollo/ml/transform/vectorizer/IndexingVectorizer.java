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

package com.gengoai.apollo.ml.transform.vectorizer;

import com.gengoai.apollo.math.linalg.NDArray;
import com.gengoai.apollo.math.linalg.NDArrayFactory;
import com.gengoai.apollo.ml.encoder.IndexEncoder;
import com.gengoai.apollo.ml.observation.Observation;
import com.gengoai.apollo.ml.observation.Sequence;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.apollo.ml.observation.VariableCollection;
import com.gengoai.conversion.Cast;
import org.apache.mahout.math.list.DoubleArrayList;

/**
 * <p>
 * A {@link Vectorizer} that outputs the index (id) of the encoded values. Works with:
 * </p>
 * <p><ul>
 * <li>{@link Variable}: variables as encoded as a scalar value representing the index of the
 * variable name</li>
 * <li>{@link VariableCollection}: encoded into a sorted NDArray of indexed variable names </li>
 * <li>{@link Sequence}: encoded into a matrix where each row is an observation in the sequence
 * and the column is the index of encoded observation at the sequence timestamp</li>
 * </ul></p>
 *
 * @author David B. Bracewell
 */
public class IndexingVectorizer extends Vectorizer {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Indexing vectorizer.
    */
   public IndexingVectorizer() {
      super(new IndexEncoder());
   }

   /**
    * Instantiates a new Indexing vectorizer.
    */
   public IndexingVectorizer(String unknownName) {
      super(new IndexEncoder(unknownName));
   }

   @Override
   public NDArray transform(Observation observation) {
      if(observation instanceof Variable) {

         int index = encoder.encode(((Variable) observation).getName());
         if(index >= 0) {
            return NDArrayFactory.ND.scalar(index);
         }
         return NDArrayFactory.ND.scalar(0);

      } else if(observation instanceof VariableCollection) {

         VariableCollection mvo = Cast.as(observation);
         DoubleArrayList list = new DoubleArrayList();
         mvo.forEach(v -> {
            int index = encoder.encode(v.getName());
            if(index >= 0) {
               list.add(index);
            }
         });
         list.sort();
         NDArray n = NDArrayFactory.ND.array(list.size());
         for(int i = 0; i < list.size(); i++) {
            n.set(i, list.get(i));
         }
         return n;

      } else if(observation instanceof Sequence) {
         Sequence<? extends Observation> sequence = Cast.as(observation);
         NDArray n = NDArrayFactory.ND.array(sequence.size());
         for(int i = 0; i < sequence.size(); i++) {
            NDArray o = transform(sequence.get(i));
            n.set(i, o.get(0));
         }
         return n;
      }
      throw new IllegalArgumentException("Unsupported Observation: " + observation.getClass());
   }

}//END OF IndexingVectorizer

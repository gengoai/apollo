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

package com.gengoai.apollo.ml.data;

import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public interface Dataset<T, E extends Dataset<T, ?>> extends Iterable<T>, Serializable, AutoCloseable {

   /**
    * <p>Iterator that provides a batch of examples per iteration.</p>
    *
    * @param batchSize the batch size
    * @return the iterator
    */
   Iterator<E> batchIterator(int batchSize);

   /**
    * Caches the examples in dataset.
    *
    * @return the cached dataset
    */
   E cache();

   /**
    * Takes the first n elements from the dataset
    *
    * @param n the number of items to take
    * @return the list of items
    */
   List<T> take(int n);

   /**
    * Creates an MStream of examples from this Dataset.
    *
    * @return the MStream of examples
    */
   MStream<T> stream();

   /**
    * Creates an MStream of examples from this Dataset.
    *
    * @return the MStream of examples
    */
   MStream<T> parallelStream();


   /**
    * Creates a new dataset containing instances from the given <code>start</code> index upto the given <code>end</code>
    * index.
    *
    * @param start the starting item index (Inclusive)
    * @param end   the ending item index (Exclusive)
    * @return the dataset
    */
   E slice(long start, long end);

   /**
    * The number of examples in the dataset
    *
    * @return the number of examples
    */
   long size();

   /**
    * Shuffles the dataset creating a new dataset.
    *
    * @return the dataset
    */
   default E shuffle() {
      return shuffle(new Random(0));
   }

   /**
    * Shuffles the dataset creating a new one with the given random number generator.
    *
    * @param random the random number generator
    * @return the dataset
    */
   E shuffle(Random random);

   /**
    * Gets the type of this dataset
    *
    * @return the {@link DatasetType}
    */
   DatasetType getType();

   /**
    * Gets a streaming context compatible with this dataset
    *
    * @return the streaming context
    */
   default StreamingContext getStreamingContext() {
      return getType().getStreamingContext();
   }

}//END OF Dataset

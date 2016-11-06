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

package com.davidbracewell.apollo.linalg;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.Iterators;
import lombok.NonNull;

import java.io.Serializable;
import java.util.*;

/**
 * <p>Provides a map interface for vectors</p>
 *
 * @author David B. Bracewell
 */
public class VectorMap extends AbstractMap<Integer, Double> implements Serializable {

   private static final long serialVersionUID = 1L;
   private final Vector vector;

   /**
    * Instantiates a new Vector map.
    *
    * @param vector the vector to wrap
    */
   private VectorMap(Vector vector) {
      this.vector = vector;
   }

   /**
    * Wraps the given vector as a map.
    *
    * @param v the vector to wrap
    * @return the map
    */
   public static Map<Integer, Double> wrap(@NonNull Vector v) {
      return new VectorMap(v);
   }

   @Override
   public Set<Entry<Integer, Double>> entrySet() {
      return new AbstractSet<Entry<Integer, Double>>() {
         @Override
         public Iterator<Entry<Integer, Double>> iterator() {
            return Iterators.transform(vector.nonZeroIterator(), e -> Tuple2.of(e.getIndex(), e.value));
         }

         @Override
         public int size() {
            return vector.size();
         }
      };
   }


   @Override
   public Set<Integer> keySet() {
      return new AbstractSet<Integer>() {

         @Override
         public boolean contains(Object o) {
            if (o instanceof Integer)
               return vector.get((Integer) o) != 0;
            return false;
         }

         @Override
         public Iterator<Integer> iterator() {
            return Iterators.transform(vector.nonZeroIterator(), Vector.Entry::getIndex);
         }

         @Override
         public int size() {
            return vector.size();
         }

      };
   }

   @Override
   public int size() {
      return vector.size();
   }

   @Override
   public boolean containsKey(Object key) {
      if (key instanceof Number) {
         return vector.get(((Number) key).intValue()) != 0;
      }
      return false;
   }

   @Override
   public Double get(Object key) {
      if (key instanceof Number) {
         return vector.get(Cast.<Number>as(key).intValue());
      }
      throw new IllegalArgumentException("Key must be a number");
   }

}//END OF VectorMap

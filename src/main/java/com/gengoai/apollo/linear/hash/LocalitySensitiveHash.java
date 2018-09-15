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

package com.gengoai.apollo.linear.hash;

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.store.VSParameter;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.multimap.HashSetMultimap;
import com.gengoai.collection.multimap.Multimap;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.math.Optimum;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

import static com.gengoai.apollo.linear.hash.LSHParameter.*;

/**
 * <p>Implementation of <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality-sensitive
 * hashing</a> which reduces high dimensional vectors into a lower k using signature functions.  The hash functions
 * cause similar vectors to be mapped into the same low k parameters.getBuckets() facilitating fast nearest neighbor
 * searches.</p>
 *
 * @author David B. Bracewell
 */
public class LocalitySensitiveHash<KEY> implements Serializable, JsonSerializable {
   private static final long LARGE_PRIME = 433494437;
   private static final long serialVersionUID = 1L;

   public Parameters<LSHParameter> getParameters() {
      return parameters;
   }

   private final Parameters<LSHParameter> parameters;
   private final SignatureFunction signatureFunction;
   private final Multimap<Integer, KEY> store = new HashSetMultimap<>();

   private static int indexHash(int x, int y) {
      return 17 + (x * 31) + y;
   }

   /**
    * Instantiates a new Locality sensitive hash.
    */
   public LocalitySensitiveHash(Parameters<LSHParameter> parameters) {
      this.parameters = parameters.copy();
      this.signatureFunction = SignatureFunction.create(parameters.get(SIGNATURE), parameters);
   }

   /**
    * Adds the given NDArray with the given NDArray id to the LSH table
    *
    * @param vector the NDArray
    */
   public void index(KEY key, NDArray vector) {
      int[] hash = hash(vector);
      for (int band = 0; band < parameters.getInt(BANDS); band++) {
         store.put(indexHash(band, hash[band]), key);
      }
   }

   private int[] booleanSignatureHash(final int[] signature) {
      long[] acc = new long[parameters.getInt(BANDS)];
      int rows = signature.length / parameters.getInt(BANDS);
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         long v = 0;
         if (signature[index] == 1) {
            v = (index + 1) * LARGE_PRIME;
         }
         int j = Math.min(index / rows, parameters.getInt(BANDS) - 1);
         acc[j] = (acc[j] + v) % Integer.MAX_VALUE;
      }

      int[] hash = new int[parameters.getInt(BANDS)];
      for (int i = 0; i < parameters.getInt(BANDS); i++) {
         hash[i] = (int) (acc[i] % parameters.getInt(BUCKETS));
      }
      return hash;
   }

   /**
    * Clears the vectors being stored.
    */
   public void clear() {
      store.clear();
   }

   /**
    * Gets the measure use to calculate the affinity between query vectors and the vectors in the table
    *
    * @return the measure
    */
   public Measure getMeasure() {
      return signatureFunction.getMeasure();
   }

   /**
    * Gets optimum associated with the LSH's measure.
    *
    * @return the optimum
    */
   public Optimum getOptimum() {
      return signatureFunction.getMeasure().getOptimum();
   }


   private int[] hash(NDArray NDArray) {
      if (signatureFunction.isBinary()) {
         return booleanSignatureHash(signatureFunction.signature(NDArray));
      }
      return intSignatureHash(signatureFunction.signature(NDArray));
   }

   private int[] intSignatureHash(final int[] signature) {
      int[] hash = new int[parameters.getInt(BANDS)];
      int rows = signature.length / parameters.getInt(BANDS);
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         int band = Math.min(index / rows, parameters.getInt(BANDS) - 1);
         hash[band] = (int) ((hash[band] + (long) signature[index] * LARGE_PRIME) % parameters.getInt(BUCKETS));
      }
      return hash;
   }

   /**
    * Finds the approximate nearest neighbors to the given input vector
    *
    * @param input the input vector
    * @return the set of nearest neighbors
    */
   public Set<KEY> query(NDArray input) {
      int[] hash = hash(input);
      Set<KEY> keys = new HashSet<>();
      for (int i = 0; i < parameters.getInt(BANDS); i++) {
         keys.addAll(store.get(indexHash(i, hash[i])));
      }
      return keys;
   }

   @Override
   public JsonEntry toJson() {
      Class<?> keyClass = store.values().stream()
                               .findFirst()
                               .map(Object::getClass)
                               .orElse(Cast.as(Object.class));
      JsonEntry lsh = JsonEntry.object()
                               .addProperty("keyClass", keyClass)
                               .addProperty("parameters", parameters.toJson());

      JsonEntry index = JsonEntry.object();
      store.keySet().forEach(i -> index.addProperty(Integer.toString(i), JsonEntry.array(store.get(i))));
      lsh.addProperty("index", index);
      return lsh;
   }

   public int size() {
      return store.size();
   }

   public static <KEY> LocalitySensitiveHash<KEY> fromJson(JsonEntry entry) {
      LocalitySensitiveHash<KEY> lsh = new LocalitySensitiveHash<>(entry.getProperty("parameters")
                                                                        .getAs(VSParameter.LSH.getValueType()));
      Class<KEY> keyClass = Cast.as(entry.getValProperty("keyClass").asClass());
      entry.getProperty("index").forEachProperty(
         (k, v) -> lsh.store.putAll(Integer.parseInt(k), v.getAsArray(keyClass)));
      return lsh;
   }

}// END OF LSH

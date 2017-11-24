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

package com.davidbracewell.apollo.hash;

import com.davidbracewell.apollo.Optimum;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.stat.measure.Measure;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * <p>Implementation of <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality-sensitive
 * hashing</a> which reduces high dimensional vectors into a lower k using signature functions.  The hash functions
 * cause similar vectors to be mapped into the same low k buckets facilitating fast nearest neighbor searches.</p>
 *
 * @author David B. Bracewell
 */
public class LocalitySensitiveHash implements Serializable {
   public static final String SIGNATURE_SIZE = "SIGNATURE_SIZE";


   private static final long LARGE_PRIME = 433494437;
   private static final long serialVersionUID = 1L;
   @Getter
   private final int bands;
   @Getter
   private final int buckets;
   @Getter
   private final int dimension;
   @Getter
   private final SignatureFunction signatureFunction;
   @Getter
   private final LSHStorage storage;

   protected LocalitySensitiveHash(int bands, int buckets, int dimension, SignatureFunction signatureFunction, LSHStorage storage) {
      this.bands = bands;
      this.buckets = buckets;
      this.dimension = dimension;
      this.signatureFunction = signatureFunction;
      this.storage = storage;
   }

   /**
    * Adds the given NDArray with the given NDArray id to the LSH table
    *
    * @param vector the NDArray
    */
   public void add(@NonNull NDArray vector) {
      int[] hash = hash(vector);
      for (int band = 0; band < bands; band++) {
         storage.add(vector, band, hash[band]);
      }
   }

   private int[] booleanSignatureHash(final int[] signature) {
      long[] acc = new long[bands];
      int rows = signature.length / bands;
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         long v = 0;
         if (signature[index] == 1) {
            v = (index + 1) * LARGE_PRIME;
         }
         int j = Math.min(index / rows, bands - 1);
         acc[j] = (acc[j] + v) % Integer.MAX_VALUE;
      }

      int[] hash = new int[bands];
      for (int i = 0; i < bands; i++) {
         hash[i] = (int) (acc[i] % buckets);
      }
      return hash;
   }

   /**
    * Clears the vectors being stored.
    */
   public void clear() {
      storage.clear();
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
      int[] hash = new int[bands];
      int rows = signature.length / bands;
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         int band = Math.min(index / rows, bands - 1);
         hash[band] = (int) ((hash[band] + (long) signature[index] * LARGE_PRIME) % buckets);
      }
      return hash;
   }

   public Set<NDArray> query(NDArray input) {
      int[] hash = hash(input);
      Set<NDArray> toReturn = new HashSet<>();
      Set<Object> keys = new HashSet<>();
      for (int i = 0; i < bands; i++) {
         storage.get(i, hash[i]).forEach(v -> {
            if (!keys.contains(v.getLabel())) {
               toReturn.add(v.copy());
               keys.addAll(v.getLabel());
            }
         });
      }
      return toReturn;
   }

   @Accessors(fluent = true)
   public static class Builder {
      @Getter
      @Setter
      private int bands = 5;
      @Getter
      @Setter
      private int buckets = 20;
      @Getter
      @Setter
      private int dimension = 100;
      @Getter
      @Setter
      private double threshold = 0.5;
      @Getter
      @Setter
      private String signature = "COSINE";

      private Map<String, Number> parameters = new HashMap<>();

      public LocalitySensitiveHash create(@NonNull LSHStorage storage) {
         if (!parameters.containsKey(SIGNATURE_SIZE)) {
            int r = (int) (Math.ceil(Math.log(1.0 / bands) / Math.log(threshold)) + 1);
            parameters.put(SIGNATURE_SIZE, r * bands);
         }
         SignatureFunction signatureFunction;
         switch (signature.toUpperCase()) {
            case "COSINE":
               signatureFunction = new CosineSignature(parameters.get(SIGNATURE_SIZE).intValue(), dimension);
               break;
            case "COSINE_DISTANCE":
               signatureFunction = new CosineDistanceSignature(parameters.get(SIGNATURE_SIZE).intValue(), dimension);
               break;
            case "EUCLIDEAN":
               signatureFunction = new EuclideanSignature(parameters.get(SIGNATURE_SIZE).intValue(),
                                                          dimension,
                                                          parameters.getOrDefault("MAXW", 100).intValue());
               break;
            case "JACCARD":
            case "MIN_HASH":
               signatureFunction = new MinHashDistanceSignature(1d - threshold, dimension);
               break;
            default:
               throw new IllegalStateException(signature + " is not one of [COSINE, COSINE_DISTANCE, EUCLIDEAN, JACCARD, MIN_HASH[");
         }

         return new LocalitySensitiveHash(bands, buckets, dimension, signatureFunction, storage);
      }

      public LocalitySensitiveHash inMemory() {
         return create(new InMemoryLSHStorage());
      }

      public Builder param(String name, Number value) {
         parameters.put(name.toUpperCase(), value);
         return this;
      }


   }


}// END OF LSH

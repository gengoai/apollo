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

package com.gengoai.apollo.hash;

import com.gengoai.apollo.hash.signature.*;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.math.Optimum;

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
   /**
    * The constant SIGNATURE_SIZE.
    */
   public static final String SIGNATURE_SIZE = "SIGNATURE_SIZE";


   private static final long LARGE_PRIME = 433494437;
   private static final long serialVersionUID = 1L;
   private final int bands;
   private final int buckets;
   private final int dimension;
   private final SignatureFunction signatureFunction;
   private final LSHStorage storage;
   private final double threshold;
   private final String signature;
   private final Map<String, Number> parameters = new HashMap<>();

   /**
    * Instantiates a new Locality sensitive hash.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param dimension         the dimension
    * @param signatureFunction the signature function
    * @param storage           the storage
    */
   private LocalitySensitiveHash(int bands, int buckets, int dimension, SignatureFunction signatureFunction, LSHStorage storage, double threshold, String signature, Map<String, Number> parameters) {
      this.bands = bands;
      this.buckets = buckets;
      this.dimension = dimension;
      this.signatureFunction = signatureFunction;
      this.storage = storage;
      this.threshold = threshold;
      this.signature = signature;
      this.parameters.putAll(parameters);
   }

   /**
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   /**
    * Adds the given NDArray with the given NDArray id to the LSH table
    *
    * @param vector the NDArray
    */
   public void add(NDArray vector) {
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
    * Gets bands.
    *
    * @return the bands
    */
   public int getBands() {
      return this.bands;
   }

   /**
    * Gets buckets.
    *
    * @return the buckets
    */
   public int getBuckets() {
      return this.buckets;
   }

   /**
    * Gets dimension.
    *
    * @return the dimension
    */
   public int getDimension() {
      return this.dimension;
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

   /**
    * Gets signature.
    *
    * @return the signature
    */
   public String getSignature() {
      return this.signature;
   }

   /**
    * Gets signature function.
    *
    * @return the signature function
    */
   public SignatureFunction getSignatureFunction() {
      return this.signatureFunction;
   }

   /**
    * Gets storage.
    *
    * @return the storage
    */
   public LSHStorage getStorage() {
      return this.storage;
   }

   /**
    * Gets threshold.
    *
    * @return the threshold
    */
   public double getThreshold() {
      return this.threshold;
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

   /**
    * Finds the approximate nearest neighbors to the given input vector
    *
    * @param input the input vector
    * @return the set of nearest neighbors
    */
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

   /**
    * To builder builder.
    *
    * @return the builder
    */
   public Builder toBuilder() {
      return builder()
                .dimension(dimension)
                .bands(bands)
                .buckets(buckets)
                .signature(signature)
                .threshold(threshold)
                .parameters(parameters);
   }

   /**
    * The type Builder.
    */
   public static class Builder {
      private int bands = 5;
      private int buckets = 20;
      private int dimension = 100;
      private double threshold = 0.5;
      private String signature = "COSINE";

      private Map<String, Number> parameters = new HashMap<>();

      /**
       * Bands builder.
       *
       * @param bands the bands
       * @return the builder
       */
      public Builder bands(int bands) {
         this.bands = bands;
         return this;
      }

      /**
       * Buckets builder.
       *
       * @param buckets the buckets
       * @return the builder
       */
      public Builder buckets(int buckets) {
         this.buckets = buckets;
         return this;
      }

      /**
       * Create locality sensitive hash.
       *
       * @param storage the storage
       * @return the locality sensitive hash
       */
      public LocalitySensitiveHash create(LSHStorage storage) {
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
               throw new IllegalStateException(
                  signature + " is not one of [COSINE, COSINE_DISTANCE, EUCLIDEAN, JACCARD, MIN_HASH[");
         }

         return new LocalitySensitiveHash(bands, buckets, dimension, signatureFunction, storage, threshold, signature,
                                          parameters);
      }

      /**
       * Dimension builder.
       *
       * @param dimension the dimension
       * @return the builder
       */
      public Builder dimension(int dimension) {
         this.dimension = dimension;
         return this;
      }

      /**
       * Gets bands.
       *
       * @return the bands
       */
      public int getBands() {
         return this.bands;
      }

      /**
       * Gets buckets.
       *
       * @return the buckets
       */
      public int getBuckets() {
         return this.buckets;
      }

      /**
       * Gets dimension.
       *
       * @return the dimension
       */
      public int getDimension() {
         return this.dimension;
      }

      /**
       * Gets signature.
       *
       * @return the signature
       */
      public String getSignature() {
         return this.signature;
      }

      /**
       * Gets threshold.
       *
       * @return the threshold
       */
      public double getThreshold() {
         return this.threshold;
      }

      /**
       * In memory locality sensitive hash.
       *
       * @return the locality sensitive hash
       */
      public LocalitySensitiveHash inMemory() {
         return create(new InMemoryLSHStorage());
      }

      /**
       * Param builder.
       *
       * @param name  the name
       * @param value the value
       * @return the builder
       */
      public Builder param(String name, Number value) {
         parameters.put(name.toUpperCase(), value);
         return this;
      }

      /**
       * Parameters builder.
       *
       * @param parameters the parameters
       * @return the builder
       */
      public Builder parameters(Map<String, Number> parameters) {
         this.parameters.putAll(parameters);
         return this;
      }

      /**
       * Signature builder.
       *
       * @param signature the signature
       * @return the builder
       */
      public Builder signature(String signature) {
         this.signature = signature;
         return this;
      }

      /**
       * Threshold builder.
       *
       * @param threshold the threshold
       * @return the builder
       */
      public Builder threshold(double threshold) {
         this.threshold = threshold;
         return this;
      }


   }


}// END OF LSH

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

import com.gengoai.apollo.hash.signature.SignatureFunction;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.multimap.HashSetMultimap;
import com.gengoai.collection.multimap.Multimap;
import com.gengoai.math.Optimum;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

/**
 * <p>Implementation of <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality-sensitive
 * hashing</a> which reduces high dimensional vectors into a lower k using signature functions.  The hash functions
 * cause similar vectors to be mapped into the same low k parameters.getBuckets() facilitating fast nearest neighbor
 * searches.</p>
 *
 * @author David B. Bracewell
 */
public class LocalitySensitiveHash2<KEY> implements Serializable {
   /**
    * The constant SIGNATURE_SIZE.
    */
   public static final String SIGNATURE_SIZE = "SIGNATURE_SIZE";


   private static final long LARGE_PRIME = 433494437;
   private static final long serialVersionUID = 1L;
   private final LSHParameters parameters;
   private final SignatureFunction signatureFunction;
   private final Multimap<Integer, KEY> store = new HashSetMultimap<>();


   private static int indexHash(int x, int y) {
      return 17 + (x * 31) + y;
   }

   /**
    * Instantiates a new Locality sensitive hash.
    */
   private LocalitySensitiveHash2(LSHParameters parameters) {
      this.parameters = parameters.copy();
      this.signatureFunction = SignatureFunction.create(parameters.getSignatureFunction(), parameters);
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
   public void index(KEY key, NDArray vector) {
      int[] hash = hash(vector);
      for (int band = 0; band < parameters.getBands(); band++) {
         store.put(indexHash(band, hash[band]), key);
      }
   }

   private int[] booleanSignatureHash(final int[] signature) {
      long[] acc = new long[parameters.getBands()];
      int rows = signature.length / parameters.getBands();
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         long v = 0;
         if (signature[index] == 1) {
            v = (index + 1) * LARGE_PRIME;
         }
         int j = Math.min(index / rows, parameters.getBands() - 1);
         acc[j] = (acc[j] + v) % Integer.MAX_VALUE;
      }

      int[] hash = new int[parameters.getBands()];
      for (int i = 0; i < parameters.getBands(); i++) {
         hash[i] = (int) (acc[i] % parameters.getBuckets());
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
    * Gets parameters.getBands().
    *
    * @return the parameters.getBands()
    */
   public int getBands() {
      return this.parameters.getBands();
   }

   /**
    * Gets parameters.getBuckets().
    *
    * @return the parameters.getBuckets()
    */
   public int getBuckets() {
      return this.parameters.getBuckets();
   }

   /**
    * Gets parameters.getDimension().
    *
    * @return the parameters.getDimension()
    */
   public int getDimension() {
      return this.parameters.getDimension();
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
      return this.parameters.getSignatureFunction();
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
    * Gets threshold.
    *
    * @return the threshold
    */
   public double getThreshold() {
      return parameters.getThreshold();
   }

   private int[] hash(NDArray NDArray) {
      if (signatureFunction.isBinary()) {
         return booleanSignatureHash(signatureFunction.signature(NDArray));
      }
      return intSignatureHash(signatureFunction.signature(NDArray));
   }

   private int[] intSignatureHash(final int[] signature) {
      int[] hash = new int[parameters.getBands()];
      int rows = signature.length / parameters.getBands();
      if (rows == 0) {
         rows = 1;
      }
      for (int index = 0; index < signature.length; index++) {
         int band = Math.min(index / rows, parameters.getBands() - 1);
         hash[band] = (int) ((hash[band] + (long) signature[index] * LARGE_PRIME) % parameters.getBuckets());
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
      for (int i = 0; i < parameters.getBands(); i++) {
         keys.addAll(store.get(indexHash(i, hash[i])));
      }
      return keys;
   }

   /**
    * To builder builder.
    *
    * @return the builder
    */
   public Builder toBuilder() {
      return builder();
   }

   /**
    * The type Builder.
    */
   public static class Builder {
//      private int parameters.
//
//      getBands() =5;
//      private int parameters.
//
//      getBuckets() =20;
//      private int parameters.
//
//      getDimension() =100;
//      private double threshold = 0.5;
//      private String signature = "COSINE";
//
//      private Map<String, Number> parameters = new HashMap<>();
//
//      /**
//       * Bands builder.
//       */
//      public Builder parameters.
//
//      getBands()(
//      int parameters.
//
//      getBands())
//
//      {
//         this.parameters.getBands() = parameters.getBands();
//         return this;
//      }
//
//      /**
//       * Buckets builder.
//       */
//      public Builder parameters.
//
//      getBuckets()(
//      int parameters.
//
//      getBuckets())
//
//      {
//         this.parameters.getBuckets() = parameters.getBuckets();
//         return this;
//      }
//
//      /**
//       * Create locality sensitive hash.
//       *
//       * @param storage the storage
//       * @return the locality sensitive hash
//       */
//      public LocalitySensitiveHash2 create(LSHStorage storage) {
//         if (!parameters.containsKey(SIGNATURE_SIZE)) {
//            int r = (int) (Math.ceil(Math.log(1.0 / parameters.getBands()) / Math.log(threshold)) + 1);
//            parameters.put(SIGNATURE_SIZE, r * parameters.getBands());
//         }
//         SignatureFunction signatureFunction;
//         switch (signature.toUpperCase()) {
//            case "COSINE":
//               signatureFunction = new CosineSignature(parameters.get(SIGNATURE_SIZE).intValue(),
//                                                       parameters.getDimension());
//               break;
//            case "COSINE_DISTANCE":
//               signatureFunction = new CosineDistanceSignature(parameters.get(SIGNATURE_SIZE).intValue(),
//                                                               parameters.getDimension());
//               break;
//            case "EUCLIDEAN":
//               signatureFunction = new EuclideanSignature(parameters.get(SIGNATURE_SIZE).intValue(),
//                                                          parameters.getDimension(),
//                                                          parameters.getOrDefault("MAXW", 100).intValue());
//               break;
//            case "JACCARD":
//            case "MIN_HASH":
//               signatureFunction = new MinHashDistanceSignature(1d - threshold, parameters.getDimension());
//               break;
//            default:
//               throw new IllegalStateException(
//                  signature + " is not one of [COSINE, COSINE_DISTANCE, EUCLIDEAN, JACCARD, MIN_HASH[");
//         }
//
//         return new LocalitySensitiveHash2(parameters.getBands(), parameters.getBuckets(), parameters.getDimension(),
//                                           signatureFunction, threshold, signature,
//                                           parameters);
//      }
//
//      /**
//       * Dimension builder.
//       */
//      public Builder parameters.
//
//      getDimension()(
//      int parameters.
//
//      getDimension())
//
//      {
//         this.parameters.getDimension() = parameters.getDimension();
//         return this;
//      }
//
//      /**
//       * Gets parameters.getBands().
//       *
//       * @return the parameters.getBands()
//       */
//      public int getBands() {
//         return this.parameters.getBands();
//      }
//
//      /**
//       * Gets parameters.getBuckets().
//       *
//       * @return the parameters.getBuckets()
//       */
//      public int getBuckets() {
//         return this.parameters.getBuckets();
//      }
//
//      /**
//       * Gets parameters.getDimension().
//       *
//       * @return the parameters.getDimension()
//       */
//      public int getDimension() {
//         return this.parameters.getDimension();
//      }
//
//      /**
//       * Gets signature.
//       *
//       * @return the signature
//       */
//      public String getSignature() {
//         return this.signature;
//      }
//
//      /**
//       * Gets threshold.
//       *
//       * @return the threshold
//       */
//      public double getThreshold() {
//         return this.threshold;
//      }
//
//      /**
//       * In memory locality sensitive hash.
//       *
//       * @return the locality sensitive hash
//       */
//      public LocalitySensitiveHash2 inMemory() {
//         return create(new InMemoryLSHStorage());
//      }
//
//      /**
//       * Param builder.
//       *
//       * @param name  the name
//       * @param value the value
//       * @return the builder
//       */
//      public Builder param(String name, Number value) {
//         parameters.put(name.toUpperCase(), value);
//         return this;
//      }
//
//      /**
//       * Parameters builder.
//       *
//       * @param parameters the parameters
//       * @return the builder
//       */
//      public Builder parameters(Map<String, Number> parameters) {
//         this.parameters.putAll(parameters);
//         return this;
//      }
//
//      /**
//       * Signature builder.
//       *
//       * @param signature the signature
//       * @return the builder
//       */
//      public Builder signature(String signature) {
//         this.signature = signature;
//         return this;
//      }
//
//      /**
//       * Threshold builder.
//       *
//       * @param threshold the threshold
//       * @return the builder
//       */
//      public Builder threshold(double threshold) {
//         this.threshold = threshold;
//         return this;
//      }


   }


}// END OF LSH

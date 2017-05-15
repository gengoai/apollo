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

package com.davidbracewell.apollo.linalg.store;

import com.davidbracewell.apollo.affinity.Measure;
import com.davidbracewell.apollo.analysis.Optimum;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.Getter;
import lombok.NonNull;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;

import java.io.Serializable;
import java.util.function.BiFunction;

/**
 * <p>Implementation of <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality-sensitive
 * hashing</a> which reduces high dimensional vectors into a lower k using signature functions.  The hash
 * functions cause similar vectors to be mapped into the same low k buckets facilitating fast nearest neighbor
 * searches.</p>
 *
 * @author David B. Bracewell
 */
public abstract class LSH implements Serializable {
   private static final long serialVersionUID = 1L;
   private static final long LARGE_PRIME = 433494437;
   @Getter
   private final int bands;
   @Getter
   private final int buckets;
   @Getter
   private final int dimension;
   @Getter
   private final SignatureFunction signatureFunction;

   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param signatureFunction the signature function
    */
   public LSH(int bands, int buckets, @NonNull SignatureFunction signatureFunction) {
      this.bands = bands;
      this.buckets = buckets;
      this.dimension = signatureFunction.getDimension();
      this.signatureFunction = signatureFunction;
   }

   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param dimension         the dimension
    * @param threshold         the threshold
    * @param signatureSupplier the signature supplier
    */
   public LSH(int bands, int buckets, int dimension, double threshold, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
      this.bands = bands;
      this.buckets = buckets;
      this.dimension = dimension;
      int r = (int) (Math.ceil(Math.log(1.0 / bands) / Math.log(threshold)) + 1);
      int signature_size = r * bands;
      this.signatureFunction = signatureSupplier.apply(signature_size, dimension);
   }

   /**
    * Instantiates a new Lsh.
    *
    * @param bands             the bands
    * @param buckets           the buckets
    * @param dimension         the dimension
    * @param signatureSupplier the signature supplier
    */
   public LSH(int bands, int buckets, int dimension, @NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
      this(bands, buckets, dimension, 0.5, signatureSupplier);
   }


   /**
    * Clears the vectors being stored.
    */
   public abstract void clear();

   /**
    * Gets the vector ids for the given band and bucket
    *
    * @param band   the band
    * @param bucket the bucket
    * @return the vector ids
    */
   protected abstract IntHashSet get(int band, int bucket);

   /**
    * Gets the ids of vectors close to the given vector
    *
    * @param vector the vector
    * @return the int ids of the nearest vectors
    */
   public IntHashSet query(@NonNull Vector vector) {
      IntHashSet matches = new IntHashSet();
      int[] hash = hash(vector);
      for (int i = 0; i < bands; i++) {
         get(i, hash[i]).forEach(matches::add);
      }
      return matches;
   }

   /**
    * Adds the given vector id to the LSH table.
    *
    * @param band   the band
    * @param bucket the bucket
    * @param vid    the vector id
    */
   protected abstract void addToTable(int band, int bucket, int vid);

   /**
    * Adds the given vector with the given vector id to the LSH table
    *
    * @param vector   the vector
    * @param vectorID the vector id
    */
   public void add(@NonNull Vector vector, int vectorID) {
      int[] hash = hash(vector);
      for (int band = 0; band < bands; band++) {
         addToTable(band, hash[band], vectorID);
      }
   }

   /**
    * Removes the vector with the given vector id from the LSH table.
    *
    * @param vector   the vector
    * @param vectorID the vector id
    */
   public void remove(Vector vector, int vectorID) {
      if (vector != null) {
         int[] hash = hash(vector);
         for (int band = 0; band < bands; band++) {
            get(band, hash[band]).remove(vectorID);
         }
      }
   }

   /**
    * Removes the vector associated with the given vector id from the LSH table.
    *
    * @param vectorID the vector id
    */
   public void remove(int vectorID) {
      for (int band = 0; band < bands; band++) {
         for (int bucket = 0; bucket < buckets; bucket++) {
            get(band, bucket).remove(vectorID);
         }
      }
   }

   private int[] hash(Vector vector) {
      if (signatureFunction.isBinary()) {
         return booleanSignatureHash(signatureFunction.signature(vector));
      }
      return intSignatureHash(signatureFunction.signature(vector));
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
    * Convenience builder for creating LSH instances.
    */
   public abstract static class Builder {
      /**
       * The Bands.
       */
      protected int bands = 5;
      /**
       * The Buckets.
       */
      protected int buckets = 20;
      /**
       * The Threshold.
       */
      protected double threshold = 0.5;
      /**
       * The Dimension.
       */
      protected int dimension = -1;
      /**
       * The Signature function.
       */
      protected SignatureFunction signatureFunction = null;
      /**
       * The Signature supplier.
       */
      protected BiFunction<Integer, Integer, SignatureFunction> signatureSupplier = CosineSignature::new;

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
       * Dimension builder.
       *
       * @param dimension the k
       * @return the builder
       */
      public Builder dimension(int dimension) {
         this.dimension = dimension;
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

      /**
       * Signature function builder.
       *
       * @param signatureFunction the signature function
       * @return the builder
       */
      public Builder signatureFunction(SignatureFunction signatureFunction) {
         this.signatureFunction = signatureFunction;
         return this;
      }

      /**
       * Signature supplier builder.
       *
       * @param signatureSupplier the signature supplier
       * @return the builder
       */
      public Builder signatureSupplier(BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
         this.signatureSupplier = signatureSupplier;
         return this;
      }

      /**
       * Create lsh.
       *
       * @return the lsh
       */
      public abstract LSH create();

      /**
       * Create vector store vector store.
       *
       * @param <KEY> the type parameter
       * @return the vector store
       */
      public abstract <KEY> VectorStore<KEY> createVectorStore();

   }


}// END OF LSH

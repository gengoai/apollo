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

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.collection.Streams;
import com.gengoai.tuple.Tuple2;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;

import static com.gengoai.tuple.Tuples.$;


/**
 * <p>Represents the results of clustering. Is treated as a {@link Model} and allows for determining the cluster of new
 * instances.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Clustering implements Model, Iterable<Cluster>, Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private final PreprocessorList<Instance> preprocessors;
   @Getter
   private final EncoderPair encoderPair;
   @Getter
   private final Measure measure;

   public Clustering(Clusterer<?> clusterer, Measure measure) {
      this.preprocessors = clusterer.getPreprocessors().getModelProcessors();
      this.encoderPair = clusterer.getEncoderPair();
      this.measure = measure;
   }

   public Clustering(Clustering other) {
      this.preprocessors = other.getPreprocessors();
      this.encoderPair = other.encoderPair;
      this.measure = other.getMeasure();
   }

   /**
    * Gets the  cluster for the given index.
    *
    * @param index the index
    * @return the cluster
    */
   public abstract Cluster get(int index);

   /**
    * Gets the root of the hierarchical cluster.
    *
    * @return the root
    */
   public Cluster getRoot() {
      throw new UnsupportedOperationException();
   }

   /**
    * Performs a hard clustering, which determines the single cluster the given instance belongs to
    *
    * @param instance the instance
    * @return the index of the cluster that the instance belongs to
    */
   public int hardCluster(@NonNull Instance instance) {
      NDArray vector = getPreprocessors().apply(instance).toVector(encoderPair);
      return getMeasure().getOptimum()
                         .optimum(Streams.asParallelStream(this)
                                         .map(c -> $(c.getId(), getMeasure().calculate(vector, c.getCentroid()))),
                                  Tuple2::getV2)
                         .map(Tuple2::getKey)
                         .orElse(-1);
   }

   /**
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   public boolean isFlat() {
      return false;
   }

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   public boolean isHierarchical() {
      return false;
   }

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   public abstract int size();

   /**
    * Performs a soft clustering, which provides a membership probability of the given instance to the clusters
    *
    * @param instance the instance
    * @return membership probability of the given instance to the clusters
    */
   public double[] softCluster(@NonNull Instance instance) {
      double[] distances = new double[size()];
      Arrays.fill(distances, Double.POSITIVE_INFINITY);
      NDArray vector = getPreprocessors().apply(instance).toVector(encoderPair);
      int assignment = hardCluster(instance);
      if (assignment >= 0) {
         distances[assignment] = getMeasure().calculate(get(assignment).getCentroid(), vector);
      }
      return distances;
   }


}//END OF Clustering

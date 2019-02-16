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

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.DiscreteModel;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.statistics.measure.Measure;
import org.apache.commons.math3.stat.descriptive.moment.Variance;

/**
 * <p>
 * Clustering is an unsupervised machine learning algorithm that partitions input objects often based on the distance
 * between them in their feature space. Different clustering algorithms may require the number of partitions (clusters)
 * to specified as a parameter whereas others may determine the optimal number of clusters automatically.
 * </p>
 * <p>
 * A clustering represents the output of fitting a {@link Clusterer} to a {@link com.gengoai.apollo.ml.data.Dataset}. It
 * provides access to the underlying clusters and provides information on whether the clustering {@link #isFlat()} or
 * {@link #isHierarchical()} along with the number of clusters ({@link #size()}).
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class Clusterer extends DiscreteModel<Clusterer> implements Iterable<Cluster> {
   private static final long serialVersionUID = 1L;
   private Measure measure;

   /**
    * Instantiates a new Clusterer.
    */
   public Clusterer(Preprocessor... preprocessors) {
      super(DiscretePipeline.unsupervised().update(p -> p.preprocessorList.addAll(preprocessors)));
   }

   /**
    * Instantiates a new Clusterer.
    *
    * @param modelParameters the model parameters
    */
   public Clusterer(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   /**
    * Gets the cluster for the given index.
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
    * Checks if the clustering is flat
    *
    * @return True if flat, False otherwise
    */
   public abstract boolean isFlat();

   /**
    * Checks if the clustering is hierarchical
    *
    * @return True if hierarchical, False otherwise
    */
   public abstract boolean isHierarchical();

   /**
    * The number of clusters
    *
    * @return the number of clusters
    */
   public abstract int size();


   /**
    * Gets the measure used to compute the distance/similarity between points.
    *
    * @return the measure
    */
   public final Measure getMeasure() {
      return measure;
   }

   protected final void setMeasure(Measure measure) {
      this.measure = measure;
   }


   /**
    * Hard cluster cluster.
    *
    * @param example the example
    * @return the cluster
    */
   public Cluster estimate(Example example) {
      return get(distances(example.transform(getPipeline())).argMax());
   }


   /**
    * Distances nd array.
    *
    * @param example the example
    * @return the nd array
    */
   public final NDArray distances(Example example){
      return distances(example.transform(getPipeline()));
   }

   /**
    * Hard cluster cluster.
    *
    * @param example the example
    * @return the cluster
    */
   public Cluster estimate(NDArray example) {
      return get(distances(example).argMax());
   }


   /**
    * Distances nd array.
    *
    * @param example the example
    * @return the nd array
    */
   public abstract NDArray distances(NDArray example);


   /**
    * Calculates the total in-group variance of the clustering.
    *
    * @return the in-group variance
    */
   public final double inGroupVariance() {
      Variance variance = new Variance();
      for (int i = 0; i < size(); i++) {
         Cluster c = get(i);
         for (NDArray point : c.getPoints()) {
            variance.increment(getMeasure().calculate(point, c.getCentroid()));
         }
      }
      return variance.getResult();
   }


   /**
    * Calculates the total between-group variance of the clustering.
    *
    * @return the between-group variance
    */
   public final double betweenGroupVariance() {
      Variance variance = new Variance();
      for (int i = 0; i < size(); i++) {
         Cluster c = get(i);
         for (int j = 0; j < size(); j++) {
            if (i != j) {
               variance.increment(getMeasure().calculate(c.getCentroid(), get(i).getCentroid()));
            }
         }
      }
      return variance.getResult();
   }

}//END OF Clusterer

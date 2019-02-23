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

import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.preprocess.Preprocessor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

/**
 * <p>Abstract base class for {@link Clusterer}s whose resultant clustering is flat, i.e. are a set of K lists of
 * points where K is the number of clusters</p>
 *
 * @author David B. Bracewell
 */
public abstract class FlatClusterer extends Clusterer {
   private static final long serialVersionUID = 1L;
   private final List<Cluster> clusters = new ArrayList<>();

   /**
    * Instantiates a new FlatClusterer model.
    *
    * @param preprocessors the preprocessors
    */
   public FlatClusterer(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new FlatClusterer model.
    *
    * @param modelParameters the model parameters
    */
   public FlatClusterer(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   /**
    * Adds a cluster to the clustering.
    *
    * @param cluster the cluster
    */
   protected void add(Cluster cluster) {
      cluster.setId(this.clusters.size());
      this.clusters.add(cluster);
   }

   @Override
   public Cluster get(int index) {
      return clusters.get(index);
   }

   @Override
   public Cluster getRoot() {
      throw new UnsupportedOperationException();
   }

   @Override
   public boolean isFlat() {
      return true;
   }

   @Override
   public boolean isHierarchical() {
      return false;
   }

   @Override
   public Iterator<Cluster> iterator() {
      return clusters.iterator();
   }

   @Override
   public int size() {
      return clusters.size();
   }


   @Override
   public Stream<Cluster> stream() {
      return clusters.stream();
   }

}// END OF FlatClustering

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

package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.guava.common.collect.HashBasedTable;
import com.davidbracewell.guava.common.collect.Table;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * <p>Clusters using a modified Chinese Restaurant Process.</p>
 *
 * @author David B. Bracewell
 */
public class CRPClusterer extends Clusterer<FlatClustering> {
   private static final Logger log = Logger.getLogger(CRPClusterer.class);
   private static final long serialVersionUID = 8347502431469266786L;
   @Getter
   @Setter
   private double alpha;
   @Getter
   private DistanceMeasure distanceMeasure;
   private Table<Vector, Vector, Double> distanceMatrix;

   /**
    * Instantiates a new CRP clusterer.
    *
    * @param distanceMeasure the distance measure
    * @param alpha           the alpha
    */
   public CRPClusterer(DistanceMeasure distanceMeasure, double alpha) {
      Preconditions.checkArgument(alpha > 0);
      this.distanceMeasure = Preconditions.checkNotNull(distanceMeasure);
      this.alpha = alpha;
   }

   @Override
   public FlatClustering cluster(@NonNull MStream<Vector> instanceStream) {
      List<Vector> instances = instanceStream.collect();
      distanceMatrix = HashBasedTable.create();
      List<Cluster> clusters = new ArrayList<>();
      clusters.add(new Cluster());
      clusters.get(0).addPoint(instances.get(0));
      Map<Vector, Integer> assignments = new HashMap<>();
      assignments.put(instances.get(0), 0);

      int report = instances.size() / 10;

      for (int i = 1; i < instances.size(); i++) {
         Vector ii = instances.get(i);

         Counter<Integer> distances = Counters.newCounter();
         for (int ci = 0; ci < clusters.size(); ci++) {
            distances.set(ci, distance(ii, clusters.get(ci)));
         }
         double sum = distances.sum();

         for (int ci = 0; ci < clusters.size(); ci++) {
            double n = (double) clusters.get(ci).size() / (i + alpha);
            distances.set(ci, n * (1d - (distances.get(ci) / sum)));
         }

         distances.set(clusters.size(), alpha / (i + alpha));
         distances.divideBySum();
         int ci = distances.sample();

         if (i % report == 0) {
            log.info("i={0}, p(new)={1}, chosen={2}, numClusters={3}", i, distances.get(clusters.size()), ci,
                     clusters.size());
         }

         while (clusters.size() <= ci) {
            clusters.add(new Cluster());
            clusters.get(clusters.size() - 1).setId(clusters.size() - 1);
         }
         clusters.get(ci).addPoint(ii);
         assignments.put(ii, ci);
      }

      int numP = instances.size() - 1;
      for (int i = 0; i < 200; i++) {
         Vector ii = instances.get((int) Math.floor(Math.random() % instances.size()));
         int cci = assignments.remove(ii);
         clusters.get(cci).getPoints().remove(ii);
         Counter<Integer> distances = Counters.newCounter();
         for (int ci = 0; ci < clusters.size(); ci++) {
            distances.set(ci, distance(ii, clusters.get(ci)));
         }
         double sum = distances.sum();

         for (int ci = 0; ci < clusters.size(); ci++) {
            double n = (double) clusters.get(ci).size() / (numP + alpha);
            distances.set(ci, n * (1d - (distances.get(ci) / sum)));
         }

         distances.set(clusters.size(), alpha / (numP + alpha));
         distances.divideBySum();
         int ci = distances.sample();
         while (clusters.size() <= ci) {
            clusters.add(new Cluster());
         }
         clusters.get(ci).addPoint(ii);
         assignments.put(ii, ci);
      }


      CRPClustering clustering = new CRPClustering(getEncoderPair(), distanceMeasure);
      clusters.stream().filter(c -> c.size() > 0).forEach(clustering::addCluster);
      return clustering;
   }

   private double distance(Vector ii, Cluster cluster) {
      double max = Double.NEGATIVE_INFINITY;
      for (Vector jj : cluster) {
         max = Math.max(max, distance(ii, jj));
      }
      return max;
   }

   private double distance(Vector ii, Vector jj) {
      if (distanceMatrix.contains(ii, jj)) {
         return distanceMatrix.get(ii, jj);
      } else if (distanceMatrix.contains(jj, ii)) {
         return distanceMatrix.get(jj, ii);
      }
      double d = distanceMeasure.calculate(ii, jj);
      distanceMatrix.put(ii, jj, d);
      return d;
   }

   /**
    * Sets the distance measure to use for clustering.
    *
    * @param distanceMeasure the distance measure
    */
   public void setDistanceMeasure(@NonNull DistanceMeasure distanceMeasure) {
      this.distanceMeasure = distanceMeasure;
   }

   @Override
   public void reset() {
      super.reset();
      this.distanceMatrix.clear();
   }

}//END OF CRPLikeClusterer

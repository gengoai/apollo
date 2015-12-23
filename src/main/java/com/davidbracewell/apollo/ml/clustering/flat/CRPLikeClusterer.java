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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterable;
import com.davidbracewell.apollo.similarity.DistanceMeasure;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.logging.Logger;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


/**
 * @author David B. Bracewell
 */
public class CRPLikeClusterer<T extends Clusterable> implements FlatClusterer<T> {
  private static final Logger log = Logger.getLogger(CRPLikeClusterer.class);
  private double alpha;
  private DistanceMeasure distanceMeasure;
  private Table<Vector, Vector, Double> distanceMatrix;

  public CRPLikeClusterer(DistanceMeasure distanceMeasure, double alpha) {
    Preconditions.checkArgument(alpha > 0);
    this.distanceMeasure = Preconditions.checkNotNull(distanceMeasure);
    this.alpha = alpha;
  }

  @Override
  public FlatClustering<T> cluster(List<? extends T> instances) {
    if (instances == null || instances.isEmpty()) {
      return new FlatClustering<>(Collections.<Cluster<T>>emptyList());
    }
    distanceMatrix = HashBasedTable.create();
    List<Cluster<T>> clusters = new ArrayList<>();
    clusters.add(new Cluster<T>());
    clusters.get(0).addPoint(instances.get(0));
    Map<T, Integer> assignments = new HashMap<>();
    assignments.put(instances.get(0), 0);

    int report = instances.size() / 10;

    for (int i = 1; i < instances.size(); i++) {
      T ii = instances.get(i);

      Counter<Integer> distances = Counters.newHashMapCounter();
      for (int ci = 0; ci < clusters.size(); ci++) {
        distances.set(ci, distance(ii.getPoint(), clusters.get(ci)));
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
        log.info("i={0}, p(new)={1}, chosen={2}, numClusters={3}", i, distances.get(clusters.size()), ci, clusters.size());
      }

      while (clusters.size() <= ci) {
        clusters.add(new Cluster<T>());
      }
      clusters.get(ci).addPoint(ii);
      assignments.put(ii, ci);
    }

    int numP = instances.size() - 1;
    for (int i = 0; i < 200; i++) {
      T ii = instances.get((int) Math.floor(Math.random() % instances.size()));
      int cci = assignments.remove(ii);
      clusters.get(cci).getPoints().remove(ii);
      Counter<Integer> distances = Counters.newHashMapCounter();
      for (int ci = 0; ci < clusters.size(); ci++) {
        distances.set(ci, distance(ii.getPoint(), clusters.get(ci)));
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
        clusters.add(new Cluster<T>());
      }
      clusters.get(ci).addPoint(ii);
      assignments.put(ii, ci);
    }


    return new FlatClustering<>(clusters.stream().filter(c -> c.size() > 0).collect(Collectors.toList()));
  }

  private double distance(Vector ii, Cluster<T> cluster) {
    double max = Double.NEGATIVE_INFINITY;
    for (T jj : cluster) {
      max = Math.max(max, distance(ii, jj.getPoint()));
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

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

}//END OF CRPLikeClusterer

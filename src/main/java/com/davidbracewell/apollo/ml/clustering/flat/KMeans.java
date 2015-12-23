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


import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterable;
import com.davidbracewell.apollo.similarity.DistanceMeasure;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class KMeans<T extends Clusterable> implements FlatClusterer<T>, Serializable {

  private static final long serialVersionUID = 1L;
  private int K;
  private int maxIterations;
  private DistanceMeasure distanceMeasure;

  public KMeans(int k, DistanceMeasure distanceMeasure) {
    this(k, distanceMeasure, Integer.MAX_VALUE);
  }

  public KMeans(int k, DistanceMeasure distanceMeasure, int maxIterations) {
    Preconditions.checkArgument(k > 0);
    Preconditions.checkArgument(maxIterations > 0);
    this.K = k;
    this.maxIterations = maxIterations;
    this.distanceMeasure = Preconditions.checkNotNull(distanceMeasure);
  }

  @Override
  public FlatClustering<T> cluster(List<? extends T> instances) {
    if (instances == null || instances.isEmpty()) {
      return new FlatClustering<>(Collections.<Cluster<T>>emptyList());
    }
    List<Cluster<T>> clusters = new ArrayList<>();
    com.davidbracewell.apollo.linalg.Vector[] centroids = initCentroids(instances);

    Map<T, Integer> assignment = new HashMap<>();


    for (int itr = 0; itr < maxIterations; itr++) {
      initClusters(clusters);
      int numMoved = 0;
      for (T ii : instances) {
        int minI = 0;
        double minD = distanceMeasure.calculate(ii.getPoint(), centroids[0]);
        for (int ci = 1; ci < K; ci++) {
          double distance = distanceMeasure.calculate(ii.getPoint(), centroids[ci]);
          if (distance < minD) {
            minD = distance;
            minI = ci;
          }
        }

        Integer old = assignment.put(ii, minI);
        clusters.get(minI).addPoint(ii);
        if (old == null || old != minI) {
          numMoved++;
        }
      }

      if (numMoved == 0) {
        break;
      }

      for (int i = 0; i < K; i++) {
        com.davidbracewell.apollo.linalg.Vector c = centroids[i];
        c.zero();
        for (Clusterable ii : clusters.get(i)) {
          c.addSelf(ii.getPoint());
        }
        c.mapDivide((double) clusters.get(i).size());
      }


    }

    return new FlatClustering<>(clusters);
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  public int getK() {
    return K;
  }

  public void setK(int k) {
    K = k;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  private com.davidbracewell.apollo.linalg.Vector[] initCentroids(List<? extends T> instances) {
    com.davidbracewell.apollo.linalg.Vector[] centroids = new com.davidbracewell.apollo.linalg.Vector[K];
    for (int i = 0; i < K; i++) {
      centroids[i] = new SparseVector(instances.get(0).getPoint().dimension());
    }

    double[] cnts = new double[K];


    Random rnd = new Random();
    for (Clusterable ii : instances) {
      int ci = rnd.nextInt(K);
      centroids[ci].addSelf(ii.getPoint());
      cnts[ci]++;
    }
    for (int i = 0; i < K; i++) {
      centroids[i].mapDivide(cnts[i]);
    }
    return centroids;
  }

  private void initClusters(List<Cluster<T>> clusters) {
    if (clusters.size() < K) {
      for (int i = 0; i < K; i++) {
        clusters.add(new Cluster<T>());
      }
    }
    for (Cluster l : clusters) {
      l.clear();
    }
  }
}//END OF KMeans

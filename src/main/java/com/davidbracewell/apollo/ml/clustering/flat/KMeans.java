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
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.*;

/**
 * @author David B. Bracewell
 */
public class KMeans extends Clusterer {
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
  public Clustering cluster(@NonNull List<FeatureVector> instances) {
    if (instances == null || instances.isEmpty()) {
      return new Clustering(getEncoderPair(), Collections.emptyList());
    }

    List<Cluster> clusters = new ArrayList<>();
    com.davidbracewell.apollo.linalg.Vector[] centroids = initCentroids(instances);

    Map<FeatureVector, Integer> assignment = new HashMap<>();


    for (int itr = 0; itr < maxIterations; itr++) {
      initClusters(clusters);
      int numMoved = 0;
      for (FeatureVector ii : instances) {
        int minI = 0;
        double minD = distanceMeasure.calculate(ii, centroids[0]);
        for (int ci = 1; ci < K; ci++) {
          double distance = distanceMeasure.calculate(ii, centroids[ci]);
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
        for (FeatureVector ii : clusters.get(i)) {
          c.addSelf(ii);
        }
        c.mapDivide((double) clusters.get(i).size());
      }


    }

    return new Clustering(getEncoderPair(), clusters);
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

  private Vector[] initCentroids(List<FeatureVector> instances) {
    Vector[] centroids = new Vector[K];
    for (int i = 0; i < K; i++) {
      centroids[i] = new SparseVector(instances.get(0).dimension());
    }

    double[] cnts = new double[K];


    Random rnd = new Random();
    for (FeatureVector ii : instances) {
      int ci = rnd.nextInt(K);
      centroids[ci].addSelf(ii);
      cnts[ci]++;
    }
    for (int i = 0; i < K; i++) {
      centroids[i].mapDivide(cnts[i]);
    }
    return centroids;
  }

  private void initClusters(List<Cluster> clusters) {
    if (clusters.size() < K) {
      for (int i = 0; i < K; i++) {
        clusters.add(new Cluster());
      }
    }
    for (Cluster l : clusters) {
      l.clear();
    }
  }
}//END OF KMeans

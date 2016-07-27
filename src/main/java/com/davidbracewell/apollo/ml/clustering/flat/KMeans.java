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


import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.MStream;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The type K means.
 *
 * @author David B. Bracewell
 */
public class KMeans extends Clusterer<FlatHardClustering> {
  private static final long serialVersionUID = 1L;
  private int K;
  private int maxIterations;
  private DistanceMeasure distanceMeasure;

  private KMeans() {
    this(1, Distance.Euclidean, 10);
  }

  /**
   * Instantiates a new K means.
   *
   * @param k               the k
   * @param distanceMeasure the distance measure
   */
  public KMeans(int k, DistanceMeasure distanceMeasure) {
    this(k, distanceMeasure, Integer.MAX_VALUE);
  }

  /**
   * Instantiates a new K means.
   *
   * @param k               the k
   * @param distanceMeasure the distance measure
   * @param maxIterations   the max iterations
   */
  public KMeans(int k, DistanceMeasure distanceMeasure, int maxIterations) {
    Preconditions.checkArgument(k > 0);
    Preconditions.checkArgument(maxIterations > 0);
    this.K = k;
    this.maxIterations = maxIterations;
    this.distanceMeasure = Preconditions.checkNotNull(distanceMeasure);
  }

  @Override
  public FlatHardClustering cluster(@NonNull MStream<LabeledVector> instanceStream) {
    FlatHardClustering clustering = new FlatHardClustering(getEncoderPair(), distanceMeasure);

    List<LabeledVector> instances = instanceStream.collect();
    for (Vector centroid : initCentroids(instances)) {
      Cluster c = new Cluster();
      c.setIndex(c.size());
      c.setCentroid(centroid);
      clustering.addCluster(c);
    }

    Map<LabeledVector, Integer> assignment = new ConcurrentHashMap<>();

    final AtomicLong numMoved = new AtomicLong(0);
    for (int itr = 0; itr < maxIterations; itr++) {
      clustering.forEach(Cluster::clear);
      numMoved.set(0);

      instances.parallelStream()
        .forEach(ii -> {
            int minI = 0;
            double minD = distanceMeasure.calculate(ii, clustering.get(0).getCentroid());
            for (int ci = 1; ci < K; ci++) {
              double distance = distanceMeasure.calculate(ii, clustering.get(ci).getCentroid());
              if (distance < minD) {
                minD = distance;
                minI = ci;
              }
            }
            Integer old = assignment.put(ii, minI);
            clustering.get(minI).addPoint(ii);
            if (old == null || old != minI) {
              numMoved.incrementAndGet();
            }
          }
        );


      if (numMoved.get() == 0) {
        break;
      }

      for (int i = 0; i < K; i++) {
        Vector c = clustering.get(i).getCentroid();
        c.zero();
        for (LabeledVector ii : clustering.get(i)) {
          c.addSelf(ii);
        }
        c.mapDivideSelf((double) clustering.get(i).size());
      }
    }

    return clustering;
  }

  /**
   * Gets distance measure.
   *
   * @return the distance measure
   */
  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  /**
   * Sets distance measure.
   *
   * @param distanceMeasure the distance measure
   */
  public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  /**
   * Gets k.
   *
   * @return the k
   */
  public int getK() {
    return K;
  }

  /**
   * Sets k.
   *
   * @param k the k
   */
  public void setK(int k) {
    K = k;
  }

  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   */
  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  private Vector[] initCentroids(List<LabeledVector> instances) {
    Vector[] centroids = new Vector[K];
    for (int i = 0; i < K; i++) {
      centroids[i] = new SparseVector(instances.get(0).dimension());
    }
    double[] cnts = new double[K];
    Random rnd = new Random();
    for (LabeledVector ii : instances) {
      int ci = rnd.nextInt(K);
      centroids[ci].addSelf(ii);
      cnts[ci]++;
    }
    for (int i = 0; i < K; i++) {
      centroids[i].mapDivideSelf(cnts[i]);
    }
    return centroids;
  }


}//END OF KMeans

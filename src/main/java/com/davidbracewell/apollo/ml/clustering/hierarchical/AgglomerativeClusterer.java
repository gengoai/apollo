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

package com.davidbracewell.apollo.ml.clustering.hierarchical;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Table;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;

/**
 * The type Agglomerative clusterer.
 *
 * @author David B. Bracewell
 */
public class AgglomerativeClusterer extends Clusterer {
  private static final long serialVersionUID = 1L;
  private DistanceMeasure distanceMeasure = Distance.Euclidean;
  private Linkage linkage = Linkage.Min;

  /**
   * Instantiates a new Agglomerative clusterer.
   */
  public AgglomerativeClusterer() {

  }

  /**
   * Instantiates a new Agglomerative clusterer.
   *
   * @param distanceMeasure the distance measure
   * @param linkage         the linkage
   */
  public AgglomerativeClusterer(DistanceMeasure distanceMeasure, Linkage linkage) {
    this.distanceMeasure = distanceMeasure;
    this.linkage = linkage;
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
   * Gets linkage.
   *
   * @return the linkage
   */
  public Linkage getLinkage() {
    return linkage;
  }

  /**
   * Sets linkage.
   *
   * @param linkage the linkage
   */
  public void setLinkage(Linkage linkage) {
    this.linkage = linkage;
  }

  @Override
  public HierarchicalClustering cluster(@NonNull List<LabeledVector> instances) {
    Table<Cluster, Cluster, Double> distanceMatrix = HashBasedTable.create();
    List<Cluster> clusters = initDistanceMatrix(instances, distanceMatrix);

    while (clusters.size() > 1) {
      doTurn(distanceMatrix, clusters);
    }

    HierarchicalClustering clustering = new HierarchicalClustering(getEncoderPair(), distanceMeasure);
    clustering.root = clusters.get(0);
    clustering.linkage = linkage;
    return clustering;
  }

  private void doTurn(Table<Cluster, Cluster, Double> distanceMatrix, List<Cluster> clusters) {
    double min = Double.POSITIVE_INFINITY;
    Tuple2<Cluster, Cluster> minC = null;
    for (int i = 0; i < clusters.size(); i++) {
      Cluster c1 = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster c2 = clusters.get(j);
        if (distanceMatrix.get(c1, c2) < min) {
          min = distanceMatrix.get(c1, c2);
          minC = Tuple2.of(c1, c2);
        }
      }
    }

    if (minC != null) {
      Cluster cprime = new Cluster();
      cprime.setLeft(minC.getV1());
      cprime.setRight(minC.getV2());
      minC.getV1().setParent(cprime);
      minC.getV2().setParent(cprime);
      cprime.setScore(min);

      distanceMatrix.row(minC.getV1()).clear();
      distanceMatrix.column(minC.getV1()).clear();
      distanceMatrix.row(minC.getV2()).clear();
      distanceMatrix.column(minC.getV2()).clear();

      clusters.remove(minC.getV1());
      clusters.remove(minC.getV2());

      for (LabeledVector point : Iterables.concat(minC.getV1().getPoints(), minC.getV2().getPoints())) {
        cprime.addPoint(point);
      }

      updateDistanceMatrix(distanceMatrix, cprime, clusters);
      clusters.add(cprime);
    }

  }

  private List<Cluster> initDistanceMatrix(List<LabeledVector> instances, Table<Cluster, Cluster, Double> distanceMatrix) {
    List<Cluster> clusters = new ArrayList<>();
    for (LabeledVector item : instances) {
      Cluster c = new Cluster();
      c.addPoint(item);
      clusters.add(c);
    }
    for (int i = 0; i < clusters.size(); i++) {
      Cluster c1 = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster c2 = clusters.get(j);
        double distance = linkage.calculate(c1, c2, distanceMeasure);
        distanceMatrix.put(c1, c2, distance);
        distanceMatrix.put(c2, c1, distance);
      }
    }
    return clusters;
  }

  private void updateDistanceMatrix(Table<Cluster, Cluster, Double> distanceMatrix, Cluster newCluster, List<Cluster> clusterList) {
    for (Cluster c1 : clusterList) {
      double d = linkage.calculate(c1, newCluster, distanceMeasure);
      distanceMatrix.put(c1, newCluster, d);
      distanceMatrix.put(newCluster, c1, d);
    }
  }


}//END OF AgglomerativeClusterer

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

import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Table;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class AgglomerativeClusterer extends Clusterer {
  private static final long serialVersionUID = 1L;
  private final DistanceMeasure distanceMeasure;
  private final Linkage linkage;

  public AgglomerativeClusterer(DistanceMeasure distanceMeasure, Linkage linkage) {
    this.distanceMeasure = distanceMeasure;
    this.linkage = linkage;
  }


  @Override
  public HierarchicalClustering cluster(@NonNull List<FeatureVector> instances) {
    Table<Cluster, Cluster, Double> distanceMatrix = HashBasedTable.create();
    List<Cluster> clusters = initDistanceMatrix(instances, distanceMatrix);

    while (clusters.size() > 1) {
      doTurn(distanceMatrix, clusters);
    }

    return new HierarchicalClustering(getEncoderPair(), Collections.singletonList(clusters.get(0)), clusters.get(0));
  }

  private double distance(Cluster c1, Cluster c2) {
    List<Double> distances = new ArrayList<>();
    double sum = 0;
    double count = 0;
    for (FeatureVector t1 : flatten(c1)) {
      for (FeatureVector t2 : flatten(c2)) {
        count++;
        distances.add(distanceMeasure.calculate(t1, t2));
        sum += distances.get(distances.size() - 1);
      }
    }
    switch (linkage) {
      case MAX:
        return Collections.max(distances);
      case MIN:
        return Collections.min(distances);
      default:
        return sum / count;
    }
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

      for (FeatureVector point : Iterables.concat(minC.getV1().getPoints(), minC.getV2().getPoints())) {
        cprime.addPoint(point);
      }

      updateDistanceMatrix(distanceMatrix, cprime, clusters);
      clusters.add(cprime);
    }

  }

  public List<FeatureVector> flatten(Cluster c) {
    if (c == null) {
      return Collections.emptyList();
    }
    if (!c.getPoints().isEmpty()) {
      return c.getPoints();
    }
    List<FeatureVector> list = new ArrayList<>();
    list.addAll(flatten(c.getLeft()));
    list.addAll(flatten(c.getRight()));
    return list;
  }

  private List<Cluster> initDistanceMatrix(List<FeatureVector> instances, Table<Cluster, Cluster, Double> distanceMatrix) {
    List<Cluster> clusters = new ArrayList<>();
    for (FeatureVector item : instances) {
      Cluster c = new Cluster();
      c.addPoint(item);
      clusters.add(c);
    }
    for (int i = 0; i < clusters.size(); i++) {
      Cluster c1 = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster c2 = clusters.get(j);
        double distance = distance(c1, c2);
        distanceMatrix.put(c1, c2, distance);
        distanceMatrix.put(c2, c1, distance);
      }
    }
    return clusters;
  }

  private void updateDistanceMatrix(Table<Cluster, Cluster, Double> distanceMatrix, Cluster newCluster, List<Cluster> clusterList) {
    for (Cluster c1 : clusterList) {
      double d = distance(c1, newCluster);
      distanceMatrix.put(c1, newCluster, d);
      distanceMatrix.put(newCluster, c1, d);
    }
  }


  public static enum Linkage {
    MAX,
    MIN,
    AVERAGE
  }


}//END OF AgglomerativeClusterer

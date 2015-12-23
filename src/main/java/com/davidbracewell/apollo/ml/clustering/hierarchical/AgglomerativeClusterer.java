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

import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterable;
import com.davidbracewell.apollo.similarity.DistanceMeasure;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Table;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class AgglomerativeClusterer<T extends Clusterable> implements HierarchicalClusterer<T>, Serializable {
  private static final long serialVersionUID = 1L;
  private final DistanceMeasure distanceMeasure;
  private final Linkage linkage;

  public AgglomerativeClusterer(DistanceMeasure distanceMeasure, Linkage linkage) {
    this.distanceMeasure = distanceMeasure;
    this.linkage = linkage;
  }


  @Override
  public HierarchicalClustering<T> cluster(List<? extends T> instances) {
    Table<Cluster<T>, Cluster<T>, Double> distanceMatrix = HashBasedTable.create();
    List<Cluster<T>> clusters = initDistanceMatrix(instances, distanceMatrix);

    while (clusters.size() > 1) {
      doTurn(distanceMatrix, clusters);
    }

    return new HierarchicalClustering<>(Arrays.asList(clusters.get(0)), clusters.get(0));
  }

  private double distance(Cluster<T> c1, Cluster<T> c2) {
    List<Double> distances = new ArrayList<>();
    double sum = 0;
    double count = 0;
    for (T t1 : flatten(c1)) {
      for (T t2 : flatten(c2)) {
        count++;
        distances.add(distanceMeasure.calculate(t1.getPoint(), t2.getPoint()));
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

  private void doTurn(Table<Cluster<T>, Cluster<T>, Double> distanceMatrix, List<Cluster<T>> clusters) {
    double min = Double.POSITIVE_INFINITY;
    Tuple2<Cluster<T>, Cluster<T>> minC = null;
    for (int i = 0; i < clusters.size(); i++) {
      Cluster<T> c1 = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster<T> c2 = clusters.get(j);
        if (distanceMatrix.get(c1, c2) < min) {
          min = distanceMatrix.get(c1, c2);
          minC = Tuple2.of(c1, c2);
        }
      }
    }

    if (minC != null) {
      Cluster<T> cprime = new Cluster<>();
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

      for (T point : Iterables.concat(minC.getV1().getPoints(), minC.getV2().getPoints())) {
        cprime.addPoint(point);
      }

      updateDistanceMatrix(distanceMatrix, cprime, clusters);
      clusters.add(cprime);
    }

  }

  public List<T> flatten(Cluster<T> c) {
    if (c == null) {
      return Collections.emptyList();
    }
    if (!c.getPoints().isEmpty()) {
      return c.getPoints();
    }
    List<T> list = new ArrayList<>();
    list.addAll(flatten(c.getLeft()));
    list.addAll(flatten(c.getRight()));
    return list;
  }

  private List<Cluster<T>> initDistanceMatrix(List<? extends T> instances, Table<Cluster<T>, Cluster<T>, Double> distanceMatrix) {
    List<Cluster<T>> clusters = new ArrayList<>();
    for (T item : instances) {
      Cluster<T> c = new Cluster<>();
      c.addPoint(item);
      clusters.add(c);
    }
    for (int i = 0; i < clusters.size(); i++) {
      Cluster<T> c1 = clusters.get(i);
      for (int j = i + 1; j < clusters.size(); j++) {
        Cluster<T> c2 = clusters.get(j);
        double distance = distance(c1, c2);
        distanceMatrix.put(c1, c2, distance);
        distanceMatrix.put(c2, c1, distance);
      }
    }
    return clusters;
  }

  private void updateDistanceMatrix(Table<Cluster<T>, Cluster<T>, Double> distanceMatrix, Cluster<T> newCluster, List<Cluster<T>> clusterList) {
    for (Cluster<T> c1 : clusterList) {
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
